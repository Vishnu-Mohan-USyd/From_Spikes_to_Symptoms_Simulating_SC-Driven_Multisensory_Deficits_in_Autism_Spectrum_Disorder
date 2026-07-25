"""Training core: ``MultiBatchAudVisMSINetworkTime`` and supporting utilities.

This module defines the multi-layer audio-visual MSI spiking network used
across all measurement scripts (TBW_test, SBW_test, EI_balance_test) and
the training driver ``run_training`` / ``train_and_save``.

Network architecture (see ``MultiBatchAudVisMSINetworkTime`` for details)
------------------------------------------------------------------------
Layers: A (auditory), V (visual), MSI excit, MSI inh, Readout.
Dynamics:
  - Conductance-based LIF with explicit AMPA + NMDA split on every
    excitatory projection (A->MSI, V->MSI, A->MSI_inh, V->MSI_inh).
  - Tsodyks-Markram short-term depression on AMPA synapses.
  - Dedicated MSI_inh -> MSI_exc GABA projection plus a direct
    inhibitory pathway from unimodal layers to MSI_exc (FFInh).
  - Lateral / surround inhibition in MSI_exc.
  - Conduction delays on A->MSI, V->MSI, MSI->Readout (default 5 substeps).
  - STDP plasticity on early layers (toggle via ``plasticity_enabled``).
  - Supervised readout training.

Substep timing
--------------
Each external "frame" / timestep is divided into ``n_substeps`` (default
100) integration substeps; with ``dt = 0.1 ms`` this gives a 10 ms outer
step and 0.1 ms inner step.

Key hyper-parameters of MultiBatchAudVisMSINetworkTime
------------------------------------------------------
  - ``pv_nmda``  (default 5.0)  — PV-cell NMDA scaling.
  - ``targ_ratio`` (default 5.0) — homeostatic target E/I ratio used by
    the AGC (automatic gain control) loop.
  - ``gNMDA``                   — global NMDA conductance gain.
  - ``g_GABA``                  — MSI_inh -> MSI_exc GABA conductance.
  - ``g_FFinh``                 — feed-forward inhibition gain (saved/
    restored by the SBW/TBW probes so adaptation state is preserved).
  - ``input_scaling`` (150.0)   — input current scaling.
  - AGC dynamics                — moving-average homeostatic adjustment of
    inhibitory gains to maintain target firing-rate balance during
    training; disabled in ``eval()`` paths used by the measurement
    scripts (``net.plasticity_enabled = False``).

Side effects
------------
At construction the network logs ``Using device: <cuda|cpu>``.  Methods
that record E/I traces or AMPA/NMDA stats keep their own buffers
(``net._ei_record``, ``net._probe``) which must be explicitly started
and stopped by the caller.

Randomness
----------
Weight initialisation uses ``torch.randn`` and ``np.random.randn`` (no
global seed pinned in this module — pin one in the calling script for
reproducibility).
"""
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize


class Positive(nn.Module):
    def forward(self, θ):
        return F.softplus(θ) - math.log(2.0)

    # optional override
    def right_inverse(self, W):
        eps = 1e-6
        return torch.log(torch.exp(W + math.log(2.0)) - 1.0 + eps)


class NonNegative(nn.Module):
    """ Strictly ≥ 0 by using plain soft‑plus (no ln2 shift). """

    def forward(self, theta):
        return F.softplus(theta)  # ≥ 0

    def right_inverse(self, W):
        eps = 1e-6
        return torch.log(torch.exp(W) - 1.0 + eps)



class AMPANMDADebugger:
    """
    Light-weight accumulator that runs silently during training.
    Activate with  net._probe = AMPANMDADebugger()
    and call   net._probe.report(epoch)   after the epoch finishes.
    """

    def __init__(self):
        self.reset()

    # --------------------------------------------------
    def reset(self):
        self.t = 0
        self.sums = defaultdict(float)

    @torch.no_grad()
    def log_EI(self, Q_exc, Q_inh):
        self.sums["Q_exc"] += Q_exc.sum().item()
        self.sums["Q_inh"] += Q_inh.sum().item()

    # --------------------------------------------------
    @torch.no_grad()
    def log(self, Q_ampa, Q_nmda, I_M, sA, sV, sM,
            R_a, R_v, mg_gate,
            J_ampa_inst, J_nmda_inst, n_spikes):  # instantaneous current args
        """
        J_ampa / J_nmda     : Tensors (B,n) – effective contributions in this sub-step
        J_ampa_inst / J_nmda_inst : Tensors (B,n) - instantaneous currents from new spikes
        I_M                 : Tensor (B,n)  – net current after all terms
        sA,sV,sM            : Spikes (B,n)
        R_a, R_v            : Tsodyks resources (B,n)
        mg_gate             : Tensor (B,n)  – Mg block factor 0…1
        """
        # current totals
        self.sums["Q_ampa"] += Q_ampa.sum().item()
        self.sums["Q_nmda"] += Q_nmda.sum().item()
        self.sums["I_exc"] += torch.clamp(I_M, min=0).sum().item()
        self.sums["I_inh"] += -torch.clamp(I_M, max=0).sum().item()

        # instant current logging
        self.sums["J_ampa_inst"] += J_ampa_inst.sum().item()
        self.sums["J_nmda_inst"] += J_nmda_inst.sum().item()
        self.sums["n_spikes"] += n_spikes

        # stats
        self.sums["spk_A"] += sA.sum().item()
        self.sums["spk_V"] += sV.sum().item()
        self.sums["spk_M"] += sM.sum().item()
        self.sums["R_a"] += R_a.mean().item()
        self.sums["R_v"] += R_v.mean().item()
        self.sums["mg"] += mg_gate.mean().item()
        self.t += 1

    # --------------------------------------------------
    def report(self, net, tag=""):
        if self.t == 0:
            print("[probe] No samples collected.")
            return
        g = self.sums  # alias
        mean = lambda k: g[k] / self.t

        print("\n──────── AMPA vs NMDA probe", tag, "────────")
        print("  Charge delivered in one epoch (∫ I dt):")
        print(f"    Q_AMPA : {g['Q_ampa']:>12.3e}")
        print(f"    Q_NMDA : {g['Q_nmda']:>12.3e}")
        print(f"    Q_NMDA / Q_AMPA : {g['Q_nmda'] / max(g['Q_ampa'], 1e-9):9.5f}")

        print("\n  Instantaneous Current Comparison (per-spike impact):")
        print(f"    J_AMPA_inst total : {g['J_ampa_inst']:>10.1f}")
        print(f"    J_NMDA_inst total : {g['J_nmda_inst']:>10.1f}")
        ratio_inst = g['J_nmda_inst'] / max(g['J_ampa_inst'], 1e-9)
        print(f"    J_NMDA_inst / J_AMPA_inst : {ratio_inst:6.3f}")
        if self.sums["n_spikes"] > 0:
            print(f"  ⟨J_AMPA⟩/spk : {mean('J_ampa_inst') / mean('n_spikes'):.3e}")

        print("\n  General Network Stats:")
        print(f"    mean E/I ratio  : {mean('I_exc') / max(mean('I_inh'), 1e-9):6.3f}")
        print(f"    Exc charge : {g['Q_exc']:>12.3e}")
        print(f"    Inh charge : {g['Q_inh']:>12.3e}")
        print(f"    E/I charge ratio : {g['Q_exc'] / max(g['Q_inh'], 1e-9):9.5f}")
        print(f"    mean R_a, R_v   : {mean('R_a'):.3f}, {mean('R_v'):.3f}")
        print(f"    mean Mg-gate    : {mean('mg'):.3f}")
        print(f"    spikes A|V|M    : {int(g['spk_A'])} , "
              f"{int(g['spk_V'])} , {int(g['spk_M'])}")
        print("────────────────────────────────────────────\n")


########################################################
#             UTILITY / GENERATION FUNCTIONS
########################################################

def location_to_index(loc_deg, n, space_size=180):
    if n <= 1:
        return 0
    frac = loc_deg / float(space_size - 1)
    return int(round(frac * (n - 1)))


def index_to_location(idx, n, space_size=180):
    if n <= 1:
        return 0
    frac = idx / float(n - 1)
    return frac * (space_size - 1)


def make_gaussian_vector_batch_gpu(center_indices, size=180, sigma=5.0, device=None):
    xs = torch.arange(size, dtype=torch.float32, device=device)
    centers = center_indices.view(-1, 1)
    dist = torch.abs(xs - centers)
    return torch.exp(-0.5 * (dist / sigma) ** 2)


def generate_event_loc_seq_batch(batch_size=32,
                                 space_size=180,
                                 offset_probability=0.1,
                                 event_duration=5,
                                 p_start=0.2,
                                 p_shift_visual=None,
                                 p_shift_audio=None,
                                 offset_range_deg=3.0,
                                 temporal_jitter_max=0):
    """
    Create synthetic A/V sequences with spatial offsets and optional temporal jitter.

    Parameters
    ----------
    temporal_jitter_max : int
        Max A/V onset offset in frames (0 disables jitter).
    """
    if p_shift_visual is None:
        p_shift_visual = offset_probability
    if p_shift_audio is None:
        p_shift_audio = offset_probability

    T = 20
    D = event_duration

    loc_seqs, mod_seqs, offs, lens = [], [], [], []

    def _merge_mode(prev: str, new: str) -> str:
        """Merge per-frame modality tags into {'A','V','B'}."""
        if prev == 'X':
            return new
        if prev == new or prev == 'B':
            return prev
        return 'B'

    for _ in range(batch_size):
        loc = [999] * T
        mode = ['X'] * T
        offA = [0.0] * T
        offV = [0.0] * T

        t = 0
        while t <= T - D:
            if np.random.rand() < p_start:
                az = int(np.random.randint(0, space_size))
                r = np.random.rand()
                mode_tag = 'B' if r < 0.6 else ('A' if r < 0.8 else 'V')
                shiftA = (np.random.rand() < p_shift_audio)
                shiftV = (np.random.rand() < p_shift_visual)
                deltaA = np.random.uniform(-offset_range_deg, offset_range_deg) if shiftA else 0.0
                deltaV = np.random.uniform(-offset_range_deg, offset_range_deg) if shiftV else 0.0

                # -------------------------------
                # temporal jitter
                # -------------------------------
                dt_frames = 0
                if mode_tag == 'B' and temporal_jitter_max and temporal_jitter_max > 0:
                    max_shift = max(0, (T - D) - t)
                    dt_frames = int(np.random.randint(-temporal_jitter_max, temporal_jitter_max + 1))
                    if abs(dt_frames) > max_shift:
                        dt_frames = int(np.sign(dt_frames) * max_shift)

                a_start = t + (-dt_frames if dt_frames < 0 else 0)
                v_start = t + (dt_frames if dt_frames > 0 else 0)
                window_len = D + abs(dt_frames)

                # audio frames
                if mode_tag in ('A', 'B'):
                    for tau in range(D):
                        idx = a_start + tau
                        if idx >= T:
                            break
                        loc[idx] = az
                        mode[idx] = _merge_mode(mode[idx], 'A')
                        offA[idx] = deltaA

                # visual frames
                if mode_tag in ('V', 'B'):
                    for tau in range(D):
                        idx = v_start + tau
                        if idx >= T:
                            break
                        loc[idx] = az
                        mode[idx] = _merge_mode(mode[idx], 'V')
                        offV[idx] = deltaV

                t += window_len
            else:
                t += 1

        loc_seqs.append(loc)
        mod_seqs.append(mode)
        offs.append({"A": offA, "V": offV})
        lens.append(T)

    return loc_seqs, mod_seqs, offs, lens




def assign_unimodal_preferred_locations(net):
    """Assign preferred location markers to unimodal A/V neurons."""

    n = net.n
    sp_size = net.space_size

    # Evenly space across map

    net.unimodal_prefA = []
    net.unimodal_prefV = []
    for i in range(n):
        locA = (i / float(n - 1)) * (sp_size - 1)
        locV = (i / float(n - 1)) * (sp_size - 1)

        net.unimodal_prefA.append(locA)
        net.unimodal_prefV.append(locV)

    print("[INFO] Assigned genetic preferred loc for each unimodal neuron (A/V).")


def assign_msi_preferred_locations(net):
    """Assign preferred location markers to MSI excitatory neurons."""

    n = net.n
    sp_size = net.space_size

    net.msi_prefA = []
    net.msi_prefV = []
    for i in range(n):
        loc_val = (i / float(n - 1)) * (sp_size - 1)

        net.msi_prefA.append(loc_val)
        net.msi_prefV.append(loc_val)

    print("[INFO] Assigned 'genetic' preferred loc for each MSI excit neuron (A->MSI, V->MSI).")


# ----------------------------------------------------------------------
# Hebbian/Oja topographic anchor
# ----------------------------------------------------------------------
def apply_topographic_anchor_unimodal(net, layer="A", lr=1e-4, sigma=5.0):
    W = net.W_inA if layer == "A" else net.W_inV  # (n,n)
    spikes = net._latest_sA if layer == "A" else net._latest_sV  # (B,n)
    r = spikes.mean(0)  # (n,)

    G = net._get_cached_gaussian_kernel(sigma)  # (n,n) — cached

    dW = lr * (r.unsqueeze(1) * G)  # Hebbian growth
    attr = "W_inA" if layer == "A" else "W_inV"
    net._p_add(attr, dW - lr * (r.unsqueeze(1) * W))  # Oja decay term


def apply_topographic_anchor_msi(net, layer="A", lr=1e-4, sigma=5.0):
    """Hebbian/Oja topographic anchoring for A/V->MSI feedforward weights."""
    # Select connection
    if layer == "A":
        W_AMPA = net.W_a2msi_AMPA
        W_NMDA = net.W_a2msi_NMDA
        spikes_in = net._latest_sA  # presyn
        # MSI spikes are post
    else:
        W_AMPA = net.W_v2msi_AMPA
        W_NMDA = net.W_v2msi_NMDA
        spikes_in = net._latest_sV

    s_post = net._latest_sMSI  # shape (B, n)
    r_post = s_post.mean(dim=0)  # average over batch => shape (n,)

    G = net._get_cached_gaussian_kernel(2.0)  # (n,n) — cached
    dW_ampa = lr * (r_post.unsqueeze(1) * G)  # shape (n,n)
    dW_ampa_decay = lr * (r_post.unsqueeze(1) * W_AMPA)
    net._p_add("W_a2msi_AMPA" if layer == "A" else "W_v2msi_AMPA", dW_ampa - dW_ampa_decay)
    dW_nmda = lr * (r_post.unsqueeze(1) * G)
    dW_nmda_decay = lr * (r_post.unsqueeze(1) * W_NMDA)
    net._p_add("W_a2msi_NMDA" if layer == "A" else "W_v2msi_NMDA", dW_nmda - dW_nmda_decay)


import torch


def decode_msi_location(
        spikes_t: torch.Tensor,
        space_size: int = 180,
        method: str = "com"
) -> torch.Tensor:
    """
    Parameters
    ----------
    spikes_t : (B, n) tensor
        Spike counts or rates of MSI excitatory neurons at one time-step
        *or* summed across the duration of an event.
    space_size : int
        Degrees represented by the map (same value you pass to
        `MultiBatchAudVisMSINetworkTime`, default 180).
    method : {"argmax", "com"}
        * "argmax": winner-take-all
        * "com"   : centre of mass
    Returns
    -------
    pred_deg : (B,) tensor
        Predicted azimuth in degrees for each item in the batch.
    """
    B, n = spikes_t.shape
    device = spikes_t.device
    idxs = torch.arange(n, device=device, dtype=torch.float32)  # 0 … n-1

    if method == "argmax":
        pred_idx = torch.argmax(spikes_t, dim=1).float()  # (B,)
    elif method == "com":
        num = torch.sum(spikes_t * idxs, dim=1)  # (B,)
        den = torch.sum(spikes_t, dim=1).clamp_min(1e-6)  # avoid /0
        pred_idx = num / den
    else:
        raise ValueError("method must be 'argmax' or 'com'")

    pred_deg = pred_idx * (space_size - 1) / (n - 1)
    return pred_deg


def decode_local_com(spikes, half_width=3):
    idx = torch.arange(spikes.size(-1), device=spikes.device)
    c = torch.argmax(spikes, -1, keepdim=True)  # peak index
    mask = (idx >= c - half_width) & (idx <= c + half_width)  # ±3 neighbours
    sp = spikes * mask
    return decode_msi_location(sp, method="com")  # same utility


def apply_local_competition_unimodal(
        net,
        layer="A",
        beta=1e-5,
        neighbor_dist=5
):
    """Local decorrelation for neighboring unimodal feedforward weights."""
    spikes = net._latest_sA if (layer == "A") else net._latest_sV
    spk_avg = spikes.mean(dim=0)  # shape (n,)

    # local range
    n = net.n
    W = net.W_inA if (layer == "A") else net.W_inV

    with torch.no_grad():
        for i in range(n):
            si = spk_avg[i].item()
            if si < 1e-9:
                continue
            # for j in [i-neighbor_dist..i+neighbor_dist], j!=i
            j_low = max(0, i - neighbor_dist)
            j_high = min(n, i + neighbor_dist + 1)
            for j in range(j_low, j_high):
                if j == i:
                    continue
                sj = spk_avg[j].item()
                if sj < 1e-9:
                    continue

                # minimal approach:
                W[i, :] -= beta * si * sj * (W[j, :] - W[i, :])
                W[j, :] -= beta * si * sj * (W[i, :] - W[j, :])


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def apply_local_competition_unimodal_fast(net,
                                          layer: str = "A",
                                          beta: float = 1e-4,
                                          neighbour_dist: int = 5,
                                          target_norm: float = None):
    """
    Lateral competition with heterosynaptic LTD **plus an L2 row clamp**.
    After subtractive LTD each row is renormalised to *target_norm*
    (default = median row-norm at call-time) so rows keep their total
    drive but are forced to concentrate on fewer presynaptic neurons.
    """
    W = net.W_inA if layer == "A" else net.W_inV
    spikes = net._latest_sA if layer == "A" else net._latest_sV
    r = spikes.mean(0)  # (n,)

    M = net._get_cached_neighbour_mask(neighbour_dist)  # (n,n) — cached
    s = torch.matmul(M, r)  # neighbour firing sum

    # heterosynaptic LTD (multiplicative)
    dW = -beta * (r * s).unsqueeze(1) * W
    net._p_add("W_inA" if layer == "A" else "W_inV", dW)

    # ---------------- L2 row-normalisation -----------------
    if target_norm is None:
        with torch.no_grad():
            target_norm = W.norm(p=2, dim=1, keepdim=True).median()

    with torch.no_grad():
        W_now = net.W_inA if layer == "A" else net.W_inV
        row_norm = W_now.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        W_now.mul_(target_norm / row_norm)


def apply_local_competition_msi_fast(net, beta=5e-4, neighbour_dist=5):
    """Heterosynaptic LTD for MSI feedforward weights (decorrelation)."""
    # MSI excit spikes
    s_msi = net._latest_sMSI  # shape (B, n)
    r = s_msi.mean(dim=0)  # shape (n,)

    M = net._get_cached_neighbour_mask(neighbour_dist)  # (n,n) — cached

    # sum of neighbor firing
    s = torch.matmul(M, r)  # shape (n,)

    d_factor = -beta * (r * s).unsqueeze(1)

    # A->MSI
    net._p_add("W_a2msi_AMPA", d_factor * net.W_a2msi_AMPA)
    net._p_add("W_a2msi_NMDA", d_factor * net.W_a2msi_NMDA)

    # V->MSI
    net._p_add("W_v2msi_AMPA", d_factor * net.W_v2msi_AMPA)
    net._p_add("W_v2msi_NMDA", d_factor * net.W_v2msi_NMDA)


def soft_row_scaling(net, target_norm=1.0, eps=1e-3):
    for attr in ("W_inA", "W_inV"):
        W = getattr(net, attr)
        with torch.no_grad():
            row_norm = W.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
            W.mul_(1.0 + eps * (target_norm / row_norm - 1.0))


# ----------------------------------------------------------------------
#  Long-timescale multiplicative scaling (biologically plausible)
# ----------------------------------------------------------------------
def slow_synaptic_scaling(W: torch.Tensor,
                          tau_hours: float = 2.0,
                          target_mean: float = 0.006,
                          dt_minutes: float = 1.0):
    """Slow homeostatic row-scaling toward target mean weight."""
    alpha = dt_minutes / (tau_hours * 60.0)
    with torch.no_grad():
        row_mean = W.mean(dim=1, keepdim=True).clamp_min(1e-9)
        scale = target_mean / row_mean
        W.mul_(1.0 + alpha * (scale - 1.0))


# task #131: REMOVED `get_target_mean(epoch_idx, ...)` — the Phase B ramp from
# task #107 drove the W_a2msi_* weights 10-26× below legacy, causing the MSI
# collapse to 0 Hz around epochs 49-53 observed in the task #128 retrain
# (debugger #130 root cause). slow_synaptic_scaling now uses the constant
# default target_mean=0.006 for all 4 MSI-input weight matrices (see the
# step_counter-gated block at ~L2520).


def generate_av_batch_tensor(
        loc_seqs,
        mod_seqs,
        offset_applied,
        n=180,
        space_size=180,
        sigma_in=5.0,
        noise_std=0.01,
        loc_jitter_std=0.0,
        stimulus_intensity=1.0,
        device=None,
        max_len=None
):
    """
    Build analog A/V inputs (batch_size, T, n) with optional spatial offsets.

    Vectorized: collects all stimulus metadata into NumPy arrays, builds ALL
    Gaussians in one batched GPU call, and scatters results back.  Identical
    output to the scalar loop when noise_std=0 and loc_jitter_std=0.

    Backwards compatibility:
      - If offset_applied[b] is a bool:
          False -> no offset (legacy).
          True  -> legacy behavior: per-frame V-only jitter (+-3 deg),
                  A remains unshifted (old code path).
      - If offset_applied[b] is a dict or (A_seq, V_seq):
          Use per-frame offsets for A and V respectively (new path).

    Parameters
    ----------
    loc_seqs : list[list[float]]
        Per-batch location sequences (degrees); 999 = no stimulus.
    mod_seqs : list[list[str]]
        Per-batch modality tags ('A', 'V', 'B', 'X').
    offset_applied : list | None
        Per-batch offset info (dict, tuple, or bool).
    n : int
        Number of spatial neurons.
    space_size : int
        Spatial extent in degrees.
    sigma_in : float
        Gaussian tuning curve width.
    noise_std : float
        Additive Gaussian noise std.
    loc_jitter_std : float
        Spatial jitter std (degrees) applied to stimulus location.
    stimulus_intensity : float
        Multiplicative scaling of Gaussian profiles.
    device : torch.device | None
        Target device.
    max_len : int | None
        Sequence length (pad shorter sequences).

    Returns
    -------
    xA_batch : Tensor (batch_size, max_len, n)
    xV_batch : Tensor (batch_size, max_len, n)
    valid_mask : Tensor (batch_size, max_len) bool
    """
    batch_size = len(loc_seqs)
    if max_len is None:
        max_len = max(len(seq) for seq in loc_seqs)

    # ── 1. Build padded NumPy arrays for metadata ─────────────────────
    loc_arr = np.full((batch_size, max_len), 999.0, dtype=np.float64)
    # Encode modality as int: X=0, A=1, V=2, B=3
    mod_enc = np.zeros((batch_size, max_len), dtype=np.int8)
    _mod_map = {'X': 0, 'A': 1, 'V': 2, 'B': 3}
    off_A = np.zeros((batch_size, max_len), dtype=np.float64)
    off_V = np.zeros((batch_size, max_len), dtype=np.float64)
    seq_lens = np.empty(batch_size, dtype=np.int64)

    for b in range(batch_size):
        T = len(loc_seqs[b])
        seq_lens[b] = T
        for t in range(T):
            loc_arr[b, t] = loc_seqs[b][t]
            mod_enc[b, t] = _mod_map.get(mod_seqs[b][t], 0)

        offs = offset_applied[b] if offset_applied is not None else False
        if isinstance(offs, dict):
            a_seq = offs.get("A", None)
            v_seq = offs.get("V", None)
            if a_seq is not None:
                la = min(T, len(a_seq))
                off_A[b, :la] = a_seq[:la]
            if v_seq is not None:
                lv = min(T, len(v_seq))
                off_V[b, :lv] = v_seq[:lv]
        elif isinstance(offs, (tuple, list)) and len(offs) == 2:
            a_seq, v_seq = offs[0], offs[1]
            if a_seq is not None:
                la = min(T, len(a_seq))
                off_A[b, :la] = a_seq[:la]
            if v_seq is not None:
                lv = min(T, len(v_seq))
                off_V[b, :lv] = v_seq[:lv]
        elif bool(offs):
            # Legacy path: random V-only jitter +-3 deg per timestep
            off_V[b, :T] = np.random.uniform(-3, 3, T)

    # ── 2. Valid mask (vectorized) ────────────────────────────────────
    idx_range = torch.arange(max_len, device=device).unsqueeze(0)        # (1, T)
    lens_t = torch.tensor(seq_lens, dtype=torch.long, device=device).unsqueeze(1)  # (B, 1)
    valid_mask = idx_range < lens_t                                       # (B, T)

    # ── 3. Active mask: not padding, not 999, not 'X' ────────────────
    active = (loc_arr != 999.0) & (mod_enc != 0)  # (B, T) numpy bool

    active_idx = np.where(active.ravel())[0]  # flat indices of active positions
    if len(active_idx) == 0:
        z = torch.zeros((batch_size, max_len, n), dtype=torch.float32, device=device)
        return z, z.clone(), valid_mask

    # ── 4. Location jitter (only active positions) ────────────────────
    if loc_jitter_std > 0.0:
        jitter = np.random.normal(0, loc_jitter_std, loc_arr.shape)
        loc_arr = np.where(active, np.clip(loc_arr + jitter, 0, space_size - 1), loc_arr)

    # ── 5. Compute neuron-index centers for A and V ───────────────────
    loc_A = loc_arr + off_A  # (B, T) degrees
    loc_V = loc_arr + off_V

    if n <= 1:
        center_A = np.zeros_like(loc_A, dtype=np.int64)
        center_V = np.zeros_like(loc_V, dtype=np.int64)
    else:
        center_A = np.rint(loc_A / (space_size - 1) * (n - 1)).astype(np.int64)
        center_V = np.rint(loc_V / (space_size - 1) * (n - 1)).astype(np.int64)

    # Gather active centres (flat) → GPU tensors
    centers_A_gpu = torch.tensor(center_A.ravel()[active_idx], dtype=torch.long, device=device)
    centers_V_gpu = torch.tensor(center_V.ravel()[active_idx], dtype=torch.long, device=device)

    # ── 6. Build ALL Gaussians in one batched call ────────────────────
    # make_gaussian_vector_batch_gpu: (K,) centres → (K, n) Gaussians
    gauss_A = make_gaussian_vector_batch_gpu(centers_A_gpu, n, sigma_in, device) * stimulus_intensity
    gauss_V = make_gaussian_vector_batch_gpu(centers_V_gpu, n, sigma_in, device) * stimulus_intensity

    # ── 7. Modality masking ───────────────────────────────────────────
    mod_active = mod_enc.ravel()[active_idx]  # (K,) int8
    # 'V'-only frames (mod_enc=2): zero A;  'A'-only frames (mod_enc=1): zero V
    mask_A = torch.tensor(mod_active != 2, dtype=torch.float32, device=device).unsqueeze(1)
    mask_V = torch.tensor(mod_active != 1, dtype=torch.float32, device=device).unsqueeze(1)
    gauss_A = gauss_A * mask_A
    gauss_V = gauss_V * mask_V

    # ── 8. Additive noise ─────────────────────────────────────────────
    if noise_std and noise_std > 0.0:
        gauss_A = gauss_A + torch.randn_like(gauss_A) * noise_std
        gauss_V = gauss_V + torch.randn_like(gauss_V) * noise_std

    # ── 9. Scatter into (B, T, n) output tensors ─────────────────────
    BT = batch_size * max_len
    xA_flat = torch.zeros((BT, n), dtype=torch.float32, device=device)
    xV_flat = torch.zeros((BT, n), dtype=torch.float32, device=device)

    scatter_idx = torch.tensor(active_idx, dtype=torch.long, device=device)
    xA_flat[scatter_idx] = gauss_A
    xV_flat[scatter_idx] = gauss_V

    xA_batch = xA_flat.view(batch_size, max_len, n)
    xV_batch = xV_flat.view(batch_size, max_len, n)

    return xA_batch, xV_batch, valid_mask


import torch
import numpy as np


def visualize_unimodal_gaussian_response_with_msi(
        net,
        center_deg=90,
        n_steps=15,
        sigma_in=5.0,
        layer="A",  # "A" or "V"
        pulse_duration=5,
        stimulus_intensity=1.0,
        device=None,
        figsize=(10, 12),
        style="seaborn-talk",
        show=True
):
    """Plot unimodal + MSI spike rasters for a Gaussian pulse."""

    if device is None:
        device = net.device

    if style is not None:
        plt.style.use(style)
    net.reset_state(batch_size=1)
    xA = torch.zeros((n_steps, net.n), dtype=torch.float32, device=device)
    xV = torch.zeros((n_steps, net.n), dtype=torch.float32, device=device)

    # Convert center_deg => index
    center_idx = int(round((center_deg / (net.space_size - 1)) * (net.n - 1)))
    center_idx = max(0, min(net.n - 1, center_idx))

    # 1D Gaussian vector
    xs = torch.arange(net.n, device=device, dtype=torch.float32)
    dist = xs - center_idx
    gauss_vec = torch.exp(-0.5 * (dist / sigma_in) ** 2) * stimulus_intensity

    if layer == "A":
        xA[:pulse_duration] = gauss_vec
    else:
        xV[:pulse_duration] = gauss_vec

    input_center_neuron = []
    for t in range(n_steps):
        if layer == "A":
            input_center_neuron.append(xA[t, center_idx].item())
        else:
            input_center_neuron.append(xV[t, center_idx].item())
    unimodal_spk_records = []  # [(t, [firing_neurons])]
    unimodal_spikes_per_t = []

    msi_spk_records = []  # same but for MSI
    msi_spikes_per_t = []

    for t in range(n_steps):
        net.update_all_layers_batch(xA[t].unsqueeze(0), xV[t].unsqueeze(0))

        # --- unimodal layer spikes ---
        if layer == "A":
            spikes_uni = net._latest_sA[0]
        else:
            spikes_uni = net._latest_sV[0]
        firing_uni = (spikes_uni > 0.5).nonzero(as_tuple=True)[0]
        unimodal_spk_records.append((t, firing_uni.detach().cpu().numpy()))
        unimodal_spikes_per_t.append(firing_uni.numel())

        # --- MSI spikes ---
        spikes_msi = net._latest_sMSI[0]
        firing_msi = (spikes_msi > 0.5).nonzero(as_tuple=True)[0]
        msi_spk_records.append((t, firing_msi.detach().cpu().numpy()))
        msi_spikes_per_t.append(firing_msi.numel())
    fig = plt.figure(figsize=figsize)

    fig.suptitle(
        f"Unimodal '{layer}' + MSI response to Gaussian\n"
        f"(center={center_deg}°, sigma={sigma_in}, pulse={pulse_duration} steps)",
        fontsize=16, fontweight='bold'
    )

    ax_input = fig.add_subplot(5, 1, 1)
    ax_input.plot(range(n_steps), input_center_neuron, marker='o', color='C0', label='Center Input')
    ax_input.set_ylabel("Input Amplitude", fontsize=12)
    ax_input.set_title("Stimulus at Center Neuron vs. Time", fontsize=12)
    ax_input.grid(True, alpha=0.3)
    ax_input.legend(loc="best")

    # -- (B) Unimodal Raster --
    ax_uni_raster = fig.add_subplot(5, 1, 2)
    all_t_uni = []
    all_idx_uni = []
    all_colors_uni = []
    for t, neuron_idxs in unimodal_spk_records:
        if len(neuron_idxs) > 0:
            all_t_uni.extend([t] * len(neuron_idxs))
            all_idx_uni.extend(neuron_idxs.tolist())
            all_colors_uni.extend(neuron_idxs.tolist())
    sc_uni = ax_uni_raster.scatter(all_t_uni, all_idx_uni, c=all_colors_uni, cmap='viridis', marker='|', s=80)
    ax_uni_raster.set_ylabel("Unimodal Neuron Index", fontsize=12)
    ax_uni_raster.set_title(f"{layer}-Layer Raster", fontsize=12)
    ax_uni_raster.set_ylim([-1, net.n])
    ax_uni_raster.grid(True, alpha=0.2)
    cb_uni = plt.colorbar(sc_uni, ax=ax_uni_raster, orientation='vertical', shrink=0.65)
    cb_uni.set_label('Neuron Index', fontsize=12)

    ax_uni_line = fig.add_subplot(5, 1, 3)
    ax_uni_line.plot(range(n_steps), unimodal_spikes_per_t, '-o', color='C1', label=f'Total Spikes ({layer})')
    ax_uni_line.set_ylabel("Spikes", fontsize=12)
    ax_uni_line.set_title(f"Unimodal '{layer}' Spikes per Time Step", fontsize=12)
    ax_uni_line.grid(True, alpha=0.3)
    ax_uni_line.legend(loc="best")

    # -- (D) MSI Raster --
    ax_msi_raster = fig.add_subplot(5, 1, 4)
    all_t_msi = []
    all_idx_msi = []
    all_colors_msi = []
    for t, neuron_idxs in msi_spk_records:
        if len(neuron_idxs) > 0:
            all_t_msi.extend([t] * len(neuron_idxs))
            all_idx_msi.extend(neuron_idxs.tolist())
            all_colors_msi.extend(neuron_idxs.tolist())
    sc_msi = ax_msi_raster.scatter(all_t_msi, all_idx_msi, c=all_colors_msi, cmap='plasma', marker='|', s=80)
    ax_msi_raster.set_ylabel("MSI Neuron Index", fontsize=12)
    ax_msi_raster.set_title("MSI Raster of Spikes", fontsize=12)
    ax_msi_raster.set_ylim([-1, net.n])
    ax_msi_raster.grid(True, alpha=0.2)
    cb_msi = plt.colorbar(sc_msi, ax=ax_msi_raster, orientation='vertical', shrink=0.65)
    cb_msi.set_label('Neuron Index', fontsize=12)

    ax_msi_line = fig.add_subplot(5, 1, 5)
    ax_msi_line.plot(range(n_steps), msi_spikes_per_t, '-o', color='C2', label='Total Spikes (MSI)')
    ax_msi_line.set_xlabel("Time step", fontsize=12)
    ax_msi_line.set_ylabel("Spikes", fontsize=12)
    ax_msi_line.set_title("MSI Spikes per Time Step", fontsize=12)
    ax_msi_line.grid(True, alpha=0.3)
    ax_msi_line.legend(loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if show:
        plt.show()

    return fig


def visualize_unimodal_gaussian_response_with_msi_rate(
        net,
        center_deg=90,
        n_steps=15,
        sigma_in=5.0,
        layer="A",  # "A" or "V"
        pulse_duration=5,
        stimulus_intensity=1.0,
        device=None,
        figsize=(10, 12),
        style="seaborn-talk",
        show=True
):
    """Plot spike rasters with colour encoding mean firing rate."""

    if device is None:
        device = net.device
    if style:
        plt.style.use(style)
    net.reset_state(batch_size=1)
    xA = torch.zeros((n_steps, net.n), device=device)
    xV = torch.zeros_like(xA)

    idx_c = int(round((center_deg / (net.space_size - 1)) * (net.n - 1)))
    idx_c = max(0, min(net.n - 1, idx_c))

    xs = torch.arange(net.n, device=device, dtype=torch.float32)
    gauss_vec = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * stimulus_intensity
    if layer == "A":
        xA[:pulse_duration] = gauss_vec
    else:
        xV[:pulse_duration] = gauss_vec
    uni_spk = torch.zeros((n_steps, net.n), device=device)
    msi_spk = torch.zeros((n_steps, net.n), device=device)
    in_amp = (xA if layer == "A" else xV)[:, idx_c].cpu().tolist()

    for t in range(n_steps):
        net.update_all_layers_batch(xA[t].unsqueeze(0), xV[t].unsqueeze(0))
        if layer == "A":
            uni_spk[t] = net._latest_sA[0]
        else:
            uni_spk[t] = net._latest_sV[0]
        msi_spk[t] = net._latest_sMSI[0]
    uni_rate = uni_spk.mean(dim=0).cpu()  # (n,)
    msi_rate = msi_spk.mean(dim=0).cpu()  # (n,)

    # avoid division-by-zero colour scaling
    eps = 1e-9
    uni_rate_norm = (uni_rate - uni_rate.min()) / (uni_rate.max() - uni_rate.min() + eps)
    msi_rate_norm = (msi_rate - msi_rate.min()) / (msi_rate.max() - msi_rate.min() + eps)

    uni_colour_map = uni_rate_norm.numpy()
    msi_colour_map = msi_rate_norm.numpy()
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"'{layer}' + MSI response | colour = mean firing-rate",
        fontsize=16, fontweight="bold"
    )

    # (A) stimulus trace
    ax_in = fig.add_subplot(5, 1, 1)
    ax_in.plot(range(n_steps), in_amp, '-o')
    ax_in.set(ylabel="Input amp.", title="Stimulus at centre neuron")
    ax_in.grid(alpha=.3)

    def build_raster_lists(spk_tensor):
        times, ids, cols = [], [], []
        for t in range(n_steps):
            active = (spk_tensor[t] > 0.5).nonzero(as_tuple=True)[0].cpu().tolist()
            if active:
                times.extend([t] * len(active))
                ids.extend(active)
        return times, ids

    # (B) unimodal raster
    t_uni, n_uni = build_raster_lists(uni_spk)
    c_uni = [uni_colour_map[i] for i in n_uni]
    ax_ru = fig.add_subplot(5, 1, 2)
    sc_u = ax_ru.scatter(t_uni, n_uni, c=c_uni, cmap="inferno", marker='|', s=80)
    ax_ru.set(ylabel=f"{layer} idx", title=f"{layer}-layer raster")
    ax_ru.set_ylim(-1, net.n);
    ax_ru.grid(alpha=.2)
    cb_u = plt.colorbar(sc_u, ax=ax_ru, shrink=.65)
    cb_u.set_label("Mean spikes/step")

    # (C) unimodal spike count over time
    ax_uc = fig.add_subplot(5, 1, 3)
    ax_uc.plot(range(n_steps), (uni_spk > 0.5).sum(1).cpu(), '-o', label="spikes / t")
    ax_uc.set(ylabel="count", title=f"Total {layer} spikes")
    ax_uc.grid(alpha=.3);
    ax_uc.legend()

    # (D) MSI raster
    t_msi, n_msi = build_raster_lists(msi_spk)
    c_msi = [msi_colour_map[i] for i in n_msi]
    ax_rm = fig.add_subplot(5, 1, 4)
    sc_m = ax_rm.scatter(t_msi, n_msi, c=c_msi, cmap="inferno", marker='|', s=80)
    ax_rm.set(ylabel="MSI idx", title="MSI raster")
    ax_rm.set_ylim(-1, net.n);
    ax_rm.grid(alpha=.2)
    cb_m = plt.colorbar(sc_m, ax=ax_rm, shrink=.65)
    cb_m.set_label("Mean spikes/step")

    # (E) MSI spike count over time
    ax_mc = fig.add_subplot(5, 1, 5)
    ax_mc.plot(range(n_steps), (msi_spk > 0.5).sum(1).cpu(), '-o', color='C2',
               label="spikes / t")
    ax_mc.set(xlabel="time-step", ylabel="count", title="Total MSI spikes")
    ax_mc.grid(alpha=.3);
    ax_mc.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if show:
        plt.show()
    return fig


# ----------------------------------------------------------------------
#  POPULATION-WIDTH METRICS & FEED-FORWARD DIAGNOSTICS
# ----------------------------------------------------------------------
def msi_pop_fwhm(spikes_1d: torch.Tensor, space_size: int = 180) -> float:
    """
    Return the full-width-at-half-maximum (degrees) of a 1-D MSI spike vector.
    Input may live on CPU or GPU; nothing is modified in-place.
    """
    spikes = spikes_1d.detach()
    if spikes.numel() <= 1:
        return 0.0
    c_idx = torch.argmax(spikes).item()
    half_peak = 0.5 * spikes[c_idx]

    l_idx = c_idx
    while l_idx > 0 and spikes[l_idx] >= half_peak:
        l_idx -= 1
    r_idx = c_idx
    n = spikes.numel()
    while r_idx < n - 1 and spikes[r_idx] >= half_peak:
        r_idx += 1

    width_neur = r_idx - l_idx
    return width_neur * (space_size - 1) / (n - 1)


def feedforward_row_stats(net, path: str = "A2MSI", sample_rows: int = 20):
    """
    Print the effective σ (neurons) of randomly sampled rows in any feed-forward
    weight matrix.
    """
    if path == "A2MSI":
        W = net.W_a2msi_AMPA + net.W_a2msi_NMDA
    elif path == "V2MSI":
        W = net.W_v2msi_AMPA + net.W_v2msi_NMDA
    elif path == "InA":
        W = net.W_inA
    elif path == "InV":
        W = net.W_inV
    else:
        raise ValueError("unknown path")

    W = W.detach().cpu()
    n = W.shape[0]
    rows = torch.linspace(0, n - 1, sample_rows).long()
    xs = torch.arange(n, dtype=torch.float32)

    print(f"[feedforward_row_stats]  path={path}")
    for i in rows:
        row = W[i]
        if row.sum() == 0:
            print(f"  row {i:3d}: EMPTY")
            continue
        mu = (row * xs).sum() / row.sum()
        var = (row * (xs - mu) ** 2).sum() / row.sum()
        print(f"  row {i:3d}   σ≈{var.sqrt():4.1f} neur.")


# -----------------------------------------------------------
# MSI activity visualization
# -----------------------------------------------------------
import matplotlib.pyplot as plt


def msi_activity_summary(
        net,
        centre_deg: float,
        sigma_in: float = 5.0,
        pulse_len: int = 6,
        n_steps: int = 25,
        modality: str = "A",  # "A" or "V"
        intensity: float = 1.0,
        style: str = "default",
        figsize=(10, 6)
):
    """Plot MSI raster, spike count per time, and spike count per neuron."""

    # ------------- switch to single batch -------------
    old_B = net.batch_size
    net.reset_state(batch_size=1)

    try:
        # ---------- build Gaussian pulse ----------
        n = net.n
        idx_c = int(round(centre_deg * (n - 1) / (net.space_size - 1)))
        xs = torch.arange(n, dtype=torch.float32, device=net.device)
        gauss = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * intensity

        xA, xV = (torch.zeros(n_steps, n, device=net.device) for _ in range(2))
        (xA if modality == "A" else xV)[:pulse_len] = gauss

        # ---------- run & record ----------
        spikes = torch.zeros(n_steps, n, device=net.device)  # MSI only
        for t in range(n_steps):
            net.update_all_layers_batch(xA[t][None, :], xV[t][None, :])
            spikes[t] = net._latest_sMSI[0]

        # ---------- prepare plots ----------
        ts_count = spikes.sum(dim=1).cpu()  # spikes per time-step  (T,)
        nu_count = spikes.sum(dim=0).cpu()  # spikes per neuron    (n,)

        plt.style.use(style)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 1], hspace=0.35)

        # -- (1) raster -----------------------------------------------------
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(spikes.cpu(),
                   cmap="Greys", aspect='auto', origin='lower', interpolation="nearest")
        ax1.set_ylabel("MSI neuron")
        ax1.set_title(f"MSI activity  |  centre={centre_deg}°, σ={sigma_in}, mode={modality}")
        ax1.axvline(0, ls="--", lw=.8, color="tab:blue")
        ax1.axvline(pulse_len, ls="--", lw=.8, color="tab:blue")

        # -- (2) spikes / time-step ----------------------------------------
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.bar(range(n_steps), ts_count, width=0.8)
        ax2.set_ylabel("Σ spikes")
        ax2.set_ylim(0, ts_count.max() * 1.1)

        # -- (3) spikes / neuron ------------------------------------------
        ax3 = fig.add_subplot(gs[2])
        ax3.bar(range(n), nu_count, width=0.8)
        ax3.set_xlabel("neuron index")
        ax3.set_ylabel("Σ spikes")
        ax3.set_xlim(0, n - 1)
        ax3.set_ylim(0, nu_count.max() * 1.1)

        plt.tight_layout()
        plt.show()

    finally:
        # restore original batch size
        if old_B != 1:
            net.reset_state(batch_size=old_B)


########################################################
#       MULTI-BATCH GPU IZHIKEVICH NETWORK CLASS
########################################################

class MultiBatchAudVisMSINetworkTime(nn.Module):
    """
    Implements a multi-layer spiking network
    (Audio, Visual, MSI excitatory, MSI inhibitory, and Readout) with:
      - A->MSI & V->MSI split into AMPA/NMDA (both excitatory).
      - A->MSI_inh & V->MSI_inh also split into AMPA/NMDA (excitatory).
      - Dedicated MSI_inh -> MSI_exc GABA projection.
      - Dedicated inhibitory projection from unimodal layers (A_inh, V_inh) directly to MSI excit.
      - Tsodyks-Markram short-term depression on AMPA synapses.
      - Conduction delays, STDP for early layers, supervised readout training.
      - Lateral (surround) inhibition in MSI excit.
    """

    def __init__(
            self,
            n_neurons=30,
            batch_size=32,
            lr_unimodal=1e-4,
            lr_msi=1e-4,
            lr_readout=1e-4,
            sigma_in=5.0,
            sigma_teacher=3.0,
            noise_std=0.1,
            single_modality_prob=0.3,
            v_thresh=0.25,
            dt=0.1,
            tau_m=5.0,
            n_substeps=100,
            loc_jitter_std=0.0,
            space_size=180,
            conduction_delay_a2msi=5,
            conduction_delay_v2msi=5,
            conduction_delay_msi2out=5
    ):
        super().__init__()
        self.n = n_neurons  # number of excitatory neurons
        self.batch_size = batch_size
        self.space_size = space_size
        self.sigma_in = sigma_in
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # teacher scheduling
        self.sigma_teacher_init = 6.0
        self.sigma_teacher_final = 2.0
        self.curriculum_epochs = 10

        self.lr_uni = lr_unimodal
        self.lr_msi = lr_msi
        self.lr_out = lr_readout
        self.noise_std = noise_std
        self.v_thresh = v_thresh
        self.dt = dt
        self.tau_m = tau_m
        self.n_substeps = n_substeps
        self.loc_jitter_std = loc_jitter_std

        self.pv_nmda = 5.0
        self.targ_ratio = 5.0   # task #128: REVERTED task #94 INT-7 (was 1.0); pristine fb6d3f6 value

        self.W_latA = torch.zeros((self.n, self.n), device=self.device)
        self.W_latV = torch.zeros((self.n, self.n), device=self.device)
        self.g_latA = 0.1  # Lateral inhibition gain for A
        self.g_latV = 0.1  # Lateral inhibition gain for V
        self._probe = AMPANMDADebugger()  # ← add near other debug fields
        self.enable_probe = False  # opt-in: set True to collect AMPA/NMDA stats
        self._ei_record = None  # E/I component recording (None = off)

        self.tau_ampa_lp = 2.5  # ms  (same as self.tau_syn)
        self.ampa_alpha = 1.0  # scale factor per injection
        self.gAMPA_LP = 1.0  # gain when converting ampa_m → current
        self.Erev_ampa = 0.0  # mV, typical AMPA reversal potential

        self.ampa_m = torch.zeros((self.batch_size, self.n),
                                  dtype=torch.float32,
                                  device=self.device)

        def pos_init(shape, scale=3.0):
            """
            Return a direct weight Parameter W with the same initial value
            that the old Positive() parametrization would expose.
            W = softplus(θ, β=1) − ln(2), where θ ~ N(0, scale²).
            """
            theta = scale * torch.randn(*shape, device=self.device)
            return nn.Parameter(Positive()(theta), requires_grad=False)

        # define an MSI inhibitory subpopulation
        self.n_inh = int(0.3 * n_neurons)
        if self.n_inh < 1:
            self.n_inh = 1

        self.input_scaling = 150.0
        self.gAMPA = 1.0  # add once in __init__
        self.Erev_ampa = 0.0  # mV, typical reversal

        # ------------- Weights: In -> Uni(A/V) --------------
        self.W_inA = pos_init((self.n, self.n), 0.1)
        self.W_inV = pos_init((self.n, self.n), 0.1)

        init_a2msi = torch.tensor(0.005 * np.random.randn(self.n, self.n),
                                  dtype=torch.float32, device=self.device)
        init_v2msi = torch.tensor(0.005 * np.random.randn(self.n, self.n),
                                  dtype=torch.float32, device=self.device)

        self.W_a2msi_AMPA = pos_init((self.n, self.n), 0.005 * 0.8)
        self.W_a2msi_NMDA = pos_init((self.n, self.n), 0.005 * 0.8)
        self.W_v2msi_AMPA = pos_init((self.n, self.n), 0.005 * 0.8)
        self.W_v2msi_NMDA = pos_init((self.n, self.n), 0.005 * 0.8)

        self.W_inA_inh = nn.Parameter(0.002 * torch.rand(self.n, self.n, device=self.device),
                                      requires_grad=False)  # U[0,0.002)  task #128: REVERTED task #80 INT-3

        self.W_inV_inh = nn.Parameter(0.002 * torch.rand(self.n, self.n, device=self.device),
                                      requires_grad=False)  # U[0,0.002)  task #128: REVERTED task #80 INT-3

        init_a2msi_inh = torch.tensor(0.005 * np.random.randn(self.n_inh, self.n),
                                      dtype=torch.float32, device=self.device)
        init_v2msi_inh = torch.tensor(0.005 * np.random.randn(self.n_inh, self.n),
                                      dtype=torch.float32, device=self.device)

        self.W_a2msiInh_AMPA = nn.Parameter(0.005 * 5.0 * torch.rand(self.n_inh, self.n, device=self.device),
                                            requires_grad=False)
        self.W_a2msiInh_NMDA = nn.Parameter(0.005 * 15.0 * torch.rand(self.n_inh, self.n, device=self.device),
                                            requires_grad=False)
        self.W_v2msiInh_AMPA = nn.Parameter(0.005 * 5.0 * torch.rand(self.n_inh, self.n, device=self.device),
                                            requires_grad=False)
        self.W_v2msiInh_NMDA = nn.Parameter(0.005 * 15.0 * torch.rand(self.n_inh, self.n, device=self.device),
                                            requires_grad=False)
        # ------------- MSI_inh -> MSI_exc (GABA) --------------

        # ------------- MSI_inh -> MSI_exc (GABA) --------------
        self.W_msiInh2Exc_GABA = nn.Parameter(0.002 * torch.rand(self.n, self.n_inh, device=self.device),
                                              requires_grad=False)  # U[0,0.002)

        self.register_buffer(
            "W_msiInh2Exc_GABA_init", self.W_msiInh2Exc_GABA.clone()
        )

        # ------------- MSI -> Out --------------
        self.W_msi2out = torch.tensor(0.01 * np.random.randn(self.n, self.n),
                                      dtype=torch.float32, device=self.device)

        # ----- inhibitory iSTDP parameters -----
        self.rho0 = 5 / 1000  # target rate per sub-step (≈5 Hz)
        self.eta_i = 1e-2  # learning-rate
        self.tau_post_i = 150.0  # decay of postsyn trace (ms)

        self.allow_inhib_plasticity = True

        self.step_counter = 0
        self.inhib_scaling_T = 2000

        # task #123: physical-time cadences for AGC fast/slow loops. These are
        # tied to physical (ms) time rather than substep count so AGC behavior
        # is dt-invariant. Reference cadences chosen to preserve dt=0.1 behavior:
        #   - fast: 0.1 ms == 1 substep at dt=0.1 (original "every substep")
        #   - slow: 10.0 ms == 100 substeps at dt=0.1 (original `% 100`)
        # task #142: reset_state() now also zeros the _last_agc_*_t fields
        # (see reset_state below). Previously these persisted across resets,
        # which froze AGC during the 2nd/3rd pass of SBW_test's AV->A->V
        # triplet (g_FFinh stuck high -> MSI suppressed -> P(fusion)=1
        # saturation). Resetting here is the within-pass correctness fix;
        # T_AGC_*_MS cadences remain unchanged.
        self.T_AGC_FAST_MS = 0.1
        self.T_AGC_SLOW_MS = 10.0
        self._last_agc_fast_t = 0.0
        self._last_agc_slow_t = 0.0

        # One trace per *excitatory* postsynaptic neuron
        self.post_i_trace = torch.zeros((self.batch_size, self.n),
                                        dtype=torch.float32,
                                        device=self.device)
        self.rate_avg_tau = 5000.0  # ms (50 s network time)
        self.post_rate_avg = torch.zeros((self.batch_size, self.n),
                                         device=self.device)

        # ------------- Biases --------------
        self.b_uniA = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.b_uniV = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.b_msi = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.b_msi_inh = torch.zeros(self.n_inh, dtype=torch.float32, device=self.device)
        self.b_out = torch.zeros(self.n, dtype=torch.float32, device=self.device)

        self.EI_history = []  # will store ratios for plots/debug

        # ------------- STDP traces --------------
        self.pre_trace_inA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_inA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_inV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_inV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_a2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_a2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_v2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_v2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_a2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.post_trace_a2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.pre_trace_v2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.post_trace_v2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)

        # ------------- Izhikevich params --------------
        # For unimodal excit
        self.aA, self.bA, self.cA, self.dA = 0.02, 0.2, -65.0, 8.0
        self.aV, self.bV, self.cV, self.dV = 0.02, 0.2, -65.0, 8.0
        # MSI excit
        self.aM, self.bM, self.cM, self.dM = 0.02, 0.2, -65.0, 8.0
        # MSI inh (fast spiking)
        self.aMi, self.bMi, self.cMi, self.dMi = 0.1, 0.2, -65.0, 2.0
        # Out
        self.aO, self.bO, self.cO, self.dO = 0.1, 0.2, -65.0, 2.0

        # ------------- Membrane potentials & recovery --------------
        # Unimodal excit
        self.v_uniA = torch.full((self.batch_size, self.n), self.cA, device=self.device)
        self.u_uniA = self.bA * self.v_uniA
        self.v_uniV = torch.full((self.batch_size, self.n), self.cV, device=self.device)
        self.u_uniV = self.bV * self.v_uniV

        # MSI excit
        self.v_msi = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.u_msi = self.bM * self.v_msi

        # MSI inh
        self.v_msi_inh = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.u_msi_inh = self.bMi * self.v_msi_inh

        # Out
        self.v_out = torch.full((self.batch_size, self.n), self.cO, device=self.device)
        self.u_out = self.bO * self.v_out

        # ------------- Spikes from last substep --------------
        self._latest_sA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self._latest_sOut = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)

        # ------------- Synaptic currents --------------
        self.tau_syn = 2.5
        self.I_A = torch.zeros((self.batch_size, self.n), device=self.device)
        self.I_V = torch.zeros((self.batch_size, self.n), device=self.device)
        self.I_M = torch.zeros((self.batch_size, self.n), device=self.device)
        self.I_M_inh = torch.zeros((self.batch_size, self.n_inh), device=self.device)
        self.I_O = torch.zeros((self.batch_size, self.n), device=self.device)

        self.I_ampa_filtered = torch.zeros((self.batch_size, self.n), device=self.device)

        # ------------- Conduction delay buffers --------------
        self.conduction_delay_a2msi = conduction_delay_a2msi
        self.conduction_delay_v2msi = conduction_delay_v2msi
        self.conduction_delay_msi2out = conduction_delay_msi2out

        # unimodal->MSI inh (dedicated inhibitory path):
        self.conduction_delay_inA_inh = conduction_delay_a2msi + 20
        self.conduction_delay_inV_inh = conduction_delay_v2msi + 20

        # unimodal->MSI_inh excit:
        self.conduction_delay_a2msi_inh = conduction_delay_a2msi + 20
        self.conduction_delay_v2msi_inh = conduction_delay_v2msi + 20

        # MSI_inh->MSI_exc
        self.conduction_delay_msi_inh2exc = max(self.conduction_delay_a2msi,
                                                self.conduction_delay_v2msi) + 50

        # task #27: physical-time conduction delays (ms). Captured at construction
        # so the physical duration is dt-invariant. Used when self.dt_correct_nmda
        # is True; buffer sizing and access then compute substep counts as
        # int(round(*_ms / self.dt)).
        self.conduction_delay_a2msi_ms       = self.conduction_delay_a2msi       * self.dt
        self.conduction_delay_v2msi_ms       = self.conduction_delay_v2msi       * self.dt
        self.conduction_delay_inA_inh_ms     = self.conduction_delay_inA_inh     * self.dt
        self.conduction_delay_inV_inh_ms     = self.conduction_delay_inV_inh     * self.dt
        self.conduction_delay_a2msi_inh_ms   = self.conduction_delay_a2msi_inh   * self.dt
        self.conduction_delay_v2msi_inh_ms   = self.conduction_delay_v2msi_inh   * self.dt
        self.conduction_delay_msi_inh2exc_ms = self.conduction_delay_msi_inh2exc * self.dt
        self.conduction_delay_msi2out_ms     = self.conduction_delay_msi2out     * self.dt

        # --- GPU ring buffers (replace Python deques) ---
        self._delay_positions = {}
        self._reset_delay_buffers()

        ################################################################
        # NMDA parameters and state variables
        ################################################################
        self.gNMDA = 0.6
        # dt-correctness flag for NMDA->I_M integration AND delays-in-ms (tasks #16/#27).
        # True (default, Stage F locked in 2026-05-16):
        #   - NMDA injection uses step-source exp-Euler at lines 2049/2138
        #     (scale source by (1 - exp(-dt/tau_syn))).
        #   - Conduction delays are derived from *_ms physical-time attributes,
        #     so substep counts scale correctly with dt at evaluation time.
        # False (legacy): preserves the original bare-add and substep-unit
        # delays for backwards bit-identity / regression testing only.
        self.dt_correct_nmda = True
        self.tau_nmda = 40.0
        self.nmda_alpha = 0.1
        self.mg_k = 0.062
        self.Erev_nmda = 10.0
        self.tau_nmdaVolt = 200.0
        self.v_nmda_rest = -65.0
        self.nmda_vrest_offset = 7.0
        self.mg_vhalf = -35.0

        self.dend_coupling_alpha = 0.1
        self.v_dend_A = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.v_dend_V = torch.full((self.batch_size, self.n), self.cM, device=self.device)

        # NMDA gating state (MSI excit)
        self.nmda_m = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.v_nmda = torch.full((self.batch_size, self.n), self.v_nmda_rest, dtype=torch.float32, device=self.device)

        self.v_dend_inhA = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.v_dend_inhV = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.nmda_m_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self.v_nmda_inh = torch.full((self.batch_size, self.n_inh), self.v_nmda_rest, dtype=torch.float32,
                                     device=self.device)

        ################################################################
        # Short-term depression (Tsodyks-Markram) for AMPA
        ################################################################
        self.R_a = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_a = torch.full((self.batch_size, self.n), 0.2, device=self.device)
        self.R_v = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_v = torch.full((self.batch_size, self.n), 0.2, device=self.device)

        self.R_a_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_a_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)
        self.R_v_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_v_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)

        self.tau_rec = 400.0
        self.tau_fac = 20.0

        # --- debug counters -------------------------------------------------
        self._dbg_spk_A = 0.0  # accumulated spikes in layer A
        self._dbg_spk_V = 0.0  # accumulated spikes in layer V
        self._dbg_spk_MSI = 0.0  # accumulated spikes in MSI excit
        self._dbg_steps = 0  # how many external frames have been seen

        # Weights are stored directly as positive/non-negative values.
        # No parametrize wrappers — clamping is done in _p_add().


        self.g_GABA = 0.7  # ① global scale  ↑  (was 0.4)

        # ── kernel cache (avoid recomputing topographic matrices) ─────
        self._kernel_cache = {}

        with torch.no_grad():
            pos = torch.arange(self.n, device=self.device)
            dist = (pos[:, None] - pos[None, :]).abs().float()

        self.register_buffer("dist_mask", dist)  # (n,n) cyclic distance

        self.R_near = 4.0  # “near” radius (neurons)
        self.eta_H = 3e-4  # Hebbian rate  (centre potentiation)
        self.eta_AH = 1e-4  # anti-Hebbian  (surround strengthening)

        self.register_buffer("near_mask", torch.exp(-(dist / self.R_near) ** 2))
        self.register_buffer("far_mask", 1.0 - self.near_mask)

        # initialise learnable surround-inhibition matrix (non-negative)
        init_W = self.near_mask.clone()
        self.W_MSI_inh = nn.Parameter(Positive()(init_W),
                                      requires_grad=False)

        self.register_buffer("W_MSI_inh_init", init_W.clone())

        self.g_GABA = 2  # was 3 – stronger Mexican-hat inhibition
        self.W_MSI_inh.mul_(1.1)  # start with higher surround weight
        self.g_FFinh = 5  # start neutral; can be auto‑calibrated
        self.freeze_g_FFinh = False  # set True during eval to prevent AGC drift
        self.plasticity_enabled = True  # set False during eval to prevent weight drift

        # self.g_GABA *= 15
        #
        # self.W_msiInh2Exc_GABA.mul_(20)
        #
        # self.W_inA_inh.mul_(20)
        # self.W_inV_inh.mul_(20)

        # ------------------------------------------------------------------
        #                       calibration helpers
        # ------------------------------------------------------------------

    def set_inhib_plasticity(self, enable: bool):
        self.allow_inhib_plasticity = enable

    def disable_all_inhibition(self):
        """
        Sets all known inhibitory pathways to zero at the raw Parameter level
        *including* reparametrized 'original' for a2msiInh/v2msiInh AMPA/NMDA.

        After this, sums of W_a2msiInh_AMPA, W_a2msiInh_NMDA, etc.
        must all be zero in the final forward pass.
        """

        def forcibly_zero_reparam(attr: str):
            if attr in self.parametrizations:  # wrapped
                plist = self.parametrizations[attr]
                theta = plist.original
                if isinstance(plist[0], NonNegative):  # cannot hit zero exactly
                    theta.fill_(-20.0)  # ≈ 2 × 10⁻⁹ after soft‑plus
                else:  # Positive → exact 0 is OK
                    theta.zero_()
            else:  # not wrapped
                w = getattr(self, attr, None)
                if w is not None:
                    w.zero_()

        with torch.no_grad():
            forcibly_zero_reparam("W_inA_inh")
            forcibly_zero_reparam("W_inV_inh")
            forcibly_zero_reparam("W_a2msiInh_AMPA")
            forcibly_zero_reparam("W_a2msiInh_NMDA")
            forcibly_zero_reparam("W_v2msiInh_AMPA")
            forcibly_zero_reparam("W_v2msiInh_NMDA")
            forcibly_zero_reparam("W_msiInh2Exc_GABA")
            forcibly_zero_reparam("W_MSI_inh")
            self.g_GABA = 0.0
            self.allow_inhib_plasticity = False

        # Now we print
        print("[INFO] All known inhibition forcibly zeroed at raw param level. Summaries:")
        print(f"  W_inA_inh sum={self.W_inA_inh.sum().item()}")
        print(f"  W_inV_inh sum={self.W_inV_inh.sum().item()}")
        print(f"  W_a2msiInh_AMPA sum={self.W_a2msiInh_AMPA.sum().item()}")
        print(f"  W_a2msiInh_NMDA sum={self.W_a2msiInh_NMDA.sum().item()}")
        print(f"  W_v2msiInh_AMPA sum={self.W_v2msiInh_AMPA.sum().item()}")
        print(f"  W_v2msiInh_NMDA sum={self.W_v2msiInh_NMDA.sum().item()}")
        print(f"  W_msiInh2Exc_GABA sum={self.W_msiInh2Exc_GABA.sum().item()}")
        print(f"  W_MSI_inh sum={self.W_MSI_inh.sum().item()}")
        print(f"  g_GABA={self.g_GABA}, allow_inhib_plasticity={self.allow_inhib_plasticity}")

    # ------------------------------------------------------------------
    @staticmethod
    def _positive_update(W: torch.Tensor, dW: torch.Tensor):
        with torch.no_grad():
            W.copy_((W + dW).clamp_(min=0.0))

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #  GPU ring-buffer helpers
    # ------------------------------------------------------------------
    def _ensure_delay_buffer(self, attr: str, *, delay: int, width: int) -> None:
        """Allocate or re-allocate a single GPU ring buffer."""
        buf_len = max(int(delay), 1)
        expected_shape = (buf_len, self.batch_size, width)
        buf = getattr(self, attr, None)
        if buf is None or tuple(buf.shape) != expected_shape:
            setattr(self, attr,
                    torch.zeros(expected_shape, dtype=torch.float32,
                                device=self.device))
        else:
            buf.zero_()
        self._delay_positions[attr] = 0

    def _delay_substeps_from_ms(self, ms: float) -> int:
        """task #27: convert a physical-ms delay to substep count at current dt."""
        return max(1, int(round(float(ms) / float(self.dt))))

    def _reset_delay_buffers(self) -> None:
        """Allocate/zero all 8 conduction-delay ring buffers.

        When self.dt_correct_nmda is True (task #27), buffer sizes are derived
        from `*_ms` physical-time attributes via the current dt, so buffers
        scale correctly when dt is changed at evaluation time.
        """
        if not hasattr(self, '_delay_positions'):
            self._delay_positions = {}
        use_ms = getattr(self, 'dt_correct_nmda', False) and \
                 hasattr(self, 'conduction_delay_a2msi_ms')
        if use_ms:
            d_a2msi       = self._delay_substeps_from_ms(self.conduction_delay_a2msi_ms)
            d_v2msi       = self._delay_substeps_from_ms(self.conduction_delay_v2msi_ms)
            d_inA_inh     = self._delay_substeps_from_ms(self.conduction_delay_inA_inh_ms)
            d_inV_inh     = self._delay_substeps_from_ms(self.conduction_delay_inV_inh_ms)
            d_a2msi_inh   = self._delay_substeps_from_ms(self.conduction_delay_a2msi_inh_ms)
            d_v2msi_inh   = self._delay_substeps_from_ms(self.conduction_delay_v2msi_inh_ms)
            d_msi_inh2exc = self._delay_substeps_from_ms(self.conduction_delay_msi_inh2exc_ms)
            d_msi2out     = self._delay_substeps_from_ms(self.conduction_delay_msi2out_ms)
        else:
            d_a2msi       = self.conduction_delay_a2msi
            d_v2msi       = self.conduction_delay_v2msi
            d_inA_inh     = self.conduction_delay_inA_inh
            d_inV_inh     = self.conduction_delay_inV_inh
            d_a2msi_inh   = self.conduction_delay_a2msi_inh
            d_v2msi_inh   = self.conduction_delay_v2msi_inh
            d_msi_inh2exc = self.conduction_delay_msi_inh2exc
            d_msi2out     = self.conduction_delay_msi2out
        self._ensure_delay_buffer("buffer_a2msi",       delay=d_a2msi,       width=self.n)
        self._ensure_delay_buffer("buffer_v2msi",       delay=d_v2msi,       width=self.n)
        self._ensure_delay_buffer("buffer_inA_inh",     delay=d_inA_inh,     width=self.n)
        self._ensure_delay_buffer("buffer_inV_inh",     delay=d_inV_inh,     width=self.n)
        self._ensure_delay_buffer("buffer_a2msi_inh",   delay=d_a2msi_inh,   width=self.n)
        self._ensure_delay_buffer("buffer_v2msi_inh",   delay=d_v2msi_inh,   width=self.n)
        self._ensure_delay_buffer("buffer_msi_inh2exc", delay=d_msi_inh2exc, width=self.n_inh)
        self._ensure_delay_buffer("buffer_msi2out",     delay=d_msi2out,     width=self.n)

    def _p_add(self, attr: str, dW: torch.Tensor,
               eps: float = 1e-9,
               rel_clip: float = 0.25,
               abs_cap: float = 5.0):
        """Clipped additive update for direct weight parameters."""
        with torch.no_grad():
            W = getattr(self, attr)
            step_limit = rel_clip * W.abs().clamp_min(eps)
            step = torch.clamp(dW, -step_limit, step_limit)
            W.copy_((W + step).clamp(min=eps, max=abs_cap))

    # ── kernel cache accessors ────────────────────────────────────────
    def _get_cached_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        """Return (n, n) Gaussian kernel, cached by sigma."""
        key = ("gauss", sigma)
        if key not in self._kernel_cache:
            idx = torch.arange(self.n, device=self.device, dtype=torch.float32)
            self._kernel_cache[key] = torch.exp(
                -0.5 * ((idx[:, None] - idx[None, :]) / sigma) ** 2
            )
        return self._kernel_cache[key]

    def _get_cached_neighbour_mask(self, dist: int) -> torch.Tensor:
        """Return (n, n) float mask where |i - j| <= dist."""
        key = ("nbr", dist)
        if key not in self._kernel_cache:
            idx = torch.arange(self.n, device=self.device, dtype=torch.float32)
            self._kernel_cache[key] = (
                (idx[:, None] - idx[None, :]).abs() <= dist
            ).float()
        return self._kernel_cache[key]

    # ------------------------------------------------------------------
    #  Legacy checkpoint support
    # ------------------------------------------------------------------
    def _translate_legacy_state_dict(self, state_dict):
        """Translate old parametrized state dicts to direct-weight format."""
        if not any(key.startswith("parametrizations.") for key in state_dict):
            return state_dict

        translated = dict(state_dict)
        positive_attrs = (
            "W_inA", "W_inV",
            "W_a2msi_AMPA", "W_a2msi_NMDA",
            "W_v2msi_AMPA", "W_v2msi_NMDA",
            "W_MSI_inh",
        )
        nonnegative_attrs = (
            "W_inA_inh", "W_inV_inh", "W_msiInh2Exc_GABA",
            "W_a2msiInh_AMPA", "W_a2msiInh_NMDA",
            "W_v2msiInh_AMPA", "W_v2msiInh_NMDA",
        )
        positive_proj = Positive()
        nonnegative_proj = NonNegative()

        for attr in positive_attrs:
            legacy_key = f"parametrizations.{attr}.original"
            if legacy_key in translated and attr not in translated:
                translated[attr] = positive_proj(translated.pop(legacy_key))
            else:
                translated.pop(legacy_key, None)

        for attr in nonnegative_attrs:
            legacy_key = f"parametrizations.{attr}.original"
            if legacy_key in translated and attr not in translated:
                translated[attr] = nonnegative_proj(translated.pop(legacy_key))
            else:
                translated.pop(legacy_key, None)

        return translated

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        translated = self._translate_legacy_state_dict(state_dict)
        return super().load_state_dict(translated, strict=strict, assign=assign)

    def _probe_spike_sum(self, *, stim_peak: float = 1.0, probe_frames: int = 15) -> float:
        """
        Deliver an audio pulse (no visual input) centred on the map and
        return the *integrated* MSI‑excit spike count produced with the *current*
        value of `self.input_scaling`.

        `probe_frames` controls the number of outer-loop frames in the probe
        (task #101: was hardcoded to 15; now configurable so callers can match
        physiologically meaningful integration windows, e.g. 100 frames).

        A healthy untrained network typically fires 200‑800 spikes here when
        `input_scaling` is in the right ball‑park (at probe_frames=15).
        """
        self.reset_state(batch_size=1)

        pulse = torch.zeros((1, self.n), device=self.device)
        pulse[0, self.n // 2] = stim_peak  # centre neuron only

        tot = 0.0
        for _ in range(probe_frames):  # task #101: parameterised (was 15)
            *_, sSum = self.update_all_layers_batch(
                pulse, torch.zeros_like(pulse),  # AUDIO‑only
                return_spike_sum=True  # <<< counts ALL sub‑steps
            )
            tot += sSum.sum().item()
        return tot

    def auto_calibrate_input_gain(self,
                                  target_MSI_spikes: int = 300,
                                  tol: int = 30,
                                  max_iter: int = 12,
                                  high_bound: float = 4000.0,
                                  probe_frames: int = 15):
        """
        Binary‑search `self.input_scaling` so that a *single‑modality* pulse
        elicits `target_MSI_spikes` ± `tol`.  Then choose the smallest
        `g_FFinh` (in 0.05 increments) that halves that response.
        """
        print(f"[CAL] calibrating input_scaling (+ g_FFinh)  "
              f"target_MSI_spikes={target_MSI_spikes}  tol={tol}  "
              f"max_iter={max_iter}  high_bound={high_bound}  "
              f"probe_frames={probe_frames}")

        # task #103: bracket the binary search with freeze_g_FFinh=True so the
        # AGC fast/slow loops (Training.py:2475-2491) do NOT drift g_FFinh during
        # the probe. Without this freeze, _probe_spike_sum mutates g_FFinh
        # between iterations, producing a non-stationary response curve and
        # poisoning the binary search. Restore the prior freeze state in finally.
        saved_freeze = getattr(self, 'freeze_g_FFinh', False)
        self.freeze_g_FFinh = True
        print(f"[CAL] AGC freeze applied (saved_freeze={saved_freeze}); search begins")

        try:
            lo, hi = 0.0, high_bound
            best_gain, best_err = self.input_scaling, float("inf")

            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            for _it in range(max_iter):
                self.input_scaling = 0.5 * (lo + hi)
                excit = self._probe_spike_sum(probe_frames=probe_frames)
                err = abs(excit - target_MSI_spikes)

                print(f"    [CAL it={_it:02d}] lo={lo:8.2f} hi={hi:8.2f} "
                      f"gain={self.input_scaling:8.2f}  spikes={excit:8.2f}  err={err:8.2f}")

                if err < best_err:
                    best_gain, best_err = self.input_scaling, err
                if err <= tol:
                    break
                if excit > target_MSI_spikes:
                    hi = self.input_scaling
                else:
                    lo = self.input_scaling

            self.input_scaling = best_gain
            converged = best_err <= tol
            print(f"[CAL] gain search done: input_scaling={self.input_scaling:.2f}  "
                  f"best_err={best_err:.2f}  converged={converged}")

            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            for g in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00, 1.50, 2.00):
                self.g_FFinh = g
                inhibited = self._probe_spike_sum(probe_frames=probe_frames)
                print(f"    [CAL gFF] g_FFinh={g:.2f}  inhibited={inhibited:8.2f}  "
                      f"(threshold={0.6*target_MSI_spikes:.2f})")
                if inhibited < 0.6 * target_MSI_spikes:
                    break  # first g that halves response

            print(f"    input_scaling={self.input_scaling:.1f}, "
                  f"g_FFinh={self.g_FFinh:.2f}")
        finally:
            # Restore the prior freeze flag so STDP / training behaves normally
            # afterwards. Calibrator KEEPS the chosen g_FFinh.
            self.freeze_g_FFinh = saved_freeze
            print(f"[CAL] AGC freeze restored to {saved_freeze}")

    def iSTDP_homeo(self, W_attr, pre_spk, post_spk, lr=1e-4, rho=0.05):
        """
        Vogels-Abbott rule.
        """
        dw = lr * torch.bmm((post_spk - rho).unsqueeze(2),
                            pre_spk.unsqueeze(1)).mean(0)

        # limit oversized RF jumps
        W_now = getattr(self, W_attr)
        dw.clamp_(-0.25 * W_now, 0.25 * W_now)  # ±25 % of current weight
        # ------------------------------------------------------------------

        self._p_add(W_attr, dw)


        with torch.no_grad():
            W = getattr(self, W_attr)
            row = W.norm(p=2, dim=1, keepdim=True).clamp_min(1e-9)
            init = getattr(self, f"{W_attr}_init").norm(p=2, dim=1, keepdim=True)
            mask = row > init
            W[mask.squeeze()] *= (init / row)[mask]

    # ── E/I component recording ──────────────────────────────────
    def start_ei_recording(self):
        """Start recording separate E/I synaptic current components."""
        self._ei_record = {
            "I_E_mean": [], "I_I_mean": [],
            "Q_E": [], "Q_I": [],
            "AMPA": [], "NMDA": [],
            "FFInh": [], "RecurInh": [], "LatInh": [],
        }

    def stop_ei_recording(self):
        """Stop recording and return dict of numpy arrays."""
        out = {k: np.asarray(v, dtype=float) for k, v in self._ei_record.items()}
        self._ei_record = None
        return out

    def reset_state(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size

        # reset unimodal
        self.v_uniA = torch.full((self.batch_size, self.n), self.cA, dtype=torch.float32, device=self.device)
        self.u_uniA = self.bA * self.v_uniA
        self.v_uniV = torch.full((self.batch_size, self.n), self.cV, dtype=torch.float32, device=self.device)
        self.u_uniV = self.bV * self.v_uniV

        # reset MSI excit
        self.v_msi = torch.full((self.batch_size, self.n), self.cM, dtype=torch.float32, device=self.device)
        self.u_msi = self.bM * self.v_msi

        # reset MSI inh
        self.v_msi_inh = torch.full((self.batch_size, self.n_inh), self.cMi, dtype=torch.float32, device=self.device)
        self.u_msi_inh = self.bMi * self.v_msi_inh

        # reset out
        self.v_out = torch.full((self.batch_size, self.n), self.cO, dtype=torch.float32, device=self.device)
        self.u_out = self.bO * self.v_out

        self.I_A = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_V = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_M = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_M_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self.I_O = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_ampa_filtered = torch.zeros((self.batch_size, self.n), device=self.device)

        self._latest_sA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self._latest_sOut = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)

        self.pre_trace_inA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_inA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_inV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_inV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_a2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_a2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_v2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_v2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_a2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.post_trace_a2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.pre_trace_v2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.post_trace_v2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.ampa_m.zero_()  # clear low-pass AMPA state

        # clear conduction ring buffers
        self._reset_delay_buffers()

        if hasattr(self, "ampa_m"):
            self.ampa_m = torch.zeros((self.batch_size, self.n),
                                      dtype=torch.float32,
                                      device=self.device)

        # reset NMDA gating
        self.nmda_m = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.v_nmda = torch.full((self.batch_size, self.n), self.v_nmda_rest, dtype=torch.float32, device=self.device)
        self.nmda_m_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self.v_nmda_inh = torch.full((self.batch_size, self.n_inh), self.v_nmda_rest, dtype=torch.float32,
                                     device=self.device)

        self.v_dend_A = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.v_dend_V = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.v_dend_inhA = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.v_dend_inhV = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)

        # reset STP
        self.R_a = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_a = torch.full((self.batch_size, self.n), 0.2, device=self.device)
        self.R_v = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_v = torch.full((self.batch_size, self.n), 0.2, device=self.device)

        self.R_a_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_a_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)
        self.R_v_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_v_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)

        self.post_i_trace = torch.zeros((self.batch_size, self.n),
                                        dtype=torch.float32,
                                        device=self.device)

        # sync firing-rate tracker
        self.post_rate_avg = torch.zeros((self.batch_size, self.n),
                                         dtype=torch.float32,
                                         device=self.device)

        # --- debug counters -------------------------------------------------
        self._dbg_spk_A = 0.0  # accumulated spikes in layer A
        self._dbg_spk_V = 0.0  # accumulated spikes in layer V
        self._dbg_spk_MSI = 0.0  # accumulated spikes in MSI excit
        self._dbg_steps = 0  # how many external frames have been seen

        # task #142: zero the AGC physical-time gates so the first frame
        # after reset_state always crosses the fast/slow cadences. Without
        # this, _last_agc_*_t carried over from a previous pass (e.g. the
        # AV leg of SBW_test's AV->A->V triplet) suppresses AGC updates in
        # the next pass -> g_FFinh stuck high -> MSI silenced -> P(fusion)
        # saturates at 1.0. Debugger-4 (task #138 E8) proved this is the
        # root cause of the SBW saturation.
        self._last_agc_fast_t = 0.0
        self._last_agc_slow_t = 0.0

        # reset RF tracking
        self.msi_rf_centers = torch.zeros((self.batch_size, self.n), device=self.device)
        self.msi_rf_certainty = torch.zeros((self.batch_size, self.n), device=self.device)

    def print_epoch_spike_summary(self, tag: str = "") -> None:
        """Print mean firing rates (Hz) for A, V, MSI layers."""
        if self._dbg_steps == 0:
            print(f"[rate] {tag} – no frames processed")
            return

        norm = self._dbg_steps * self.n  # total neuron-frames
        rA = self._dbg_spk_A / norm
        rV = self._dbg_spk_V / norm
        rM = self._dbg_spk_MSI / norm

        # Convert to Hz
        ms_per_frame = self.n_substeps * self.dt
        hz_fact = 1000.0 / ms_per_frame
        rA_hz, rV_hz, rM_hz = (x * hz_fact for x in (rA, rV, rM))

        print(f"[rate] {tag:10s}"
              f"  A={rA_hz:6.2f} Hz"
              f"  V={rV_hz:6.2f} Hz"
              f"  MSI={rM_hz:6.2f} Hz"
              f"   (target={self.rho0 * hz_fact:5.2f} Hz)")

        # ready for next epoch
        self._dbg_spk_A = self._dbg_spk_V = self._dbg_spk_MSI = 0.0
        self._dbg_steps = 0

    def update_all_layers_batch(self,
                                xA_batch,
                                xV_batch,
                                valid_mask=None,
                                record_voltages=False,
                                debug=False,
                                conduction_debug=False,
                                curr_debug=False,
                                epoch_idx=0,
                                return_delayed=False,
                                return_spike_sum=False):  # optional spike-sum
        """
        Forward-prop one external time-step (100 Izhikevich sub-steps).

        If `return_spike_sum` is True, an extra tensor
            sum_sM   (batch_size , n)
        containing the **total number of MSI spikes in this external frame**
        is appended to the return tuple.
        """

        batch_size = xA_batch.size(0)

        if return_spike_sum:
            sum_sM = torch.zeros(batch_size, self.n, device=self.device)

        if valid_mask is not None:
            mask = valid_mask.view(batch_size, 1)
            xA_batch = xA_batch * mask
            xV_batch = xV_batch * mask

        sA = torch.zeros((batch_size, self.n), dtype=torch.float32, device=self.device)
        sV = torch.zeros((batch_size, self.n), dtype=torch.float32, device=self.device)
        sM = torch.zeros((batch_size, self.n), dtype=torch.float32, device=self.device)
        sMi = torch.zeros((batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        sO = torch.zeros((batch_size, self.n), dtype=torch.float32, device=self.device)

        decay_factor = 1.0 - self.dt / self.tau_syn
        ampa_decay = 1.0 - self.dt / self.tau_ampa_lp
        nmda_decay = 1.0 - self.dt / self.tau_nmda
        # Per-step source-onto-decaying-state scale (task #16/#121/#123/#135).
        #   dt_linear_scale: linear `dt / 0.1` applied to the AMPA and AGC/FF-inh
        #     AMPA injection paths. At canonical dt=0.1 this is 1.0 (byte-identical
        #     to bare add → legacy ckpts produce paper biology unchanged). At other
        #     dt it scales per-substep injection so total per-ms injection is
        #     preserved → dt-invariant.
        #   task #147: the NMDA->I_M injection was REVERTED off dt_linear_scale back
        #     to the exp-Euler form below. task #135's ×1.0 NMDA scaling was a ~25×
        #     over-drive at the forced gNMDA=1.30 operating point (paper-tuned drive
        #     is 1.30 × 0.0392 ≈ 0.05), which flipped MS enhancement negative and
        #     over-widened TBW. See scratchpad/06_regression_diagnosis.md.
        dt_linear_scale = self.dt / 0.1
        # Per-step NMDA->I_M source scale for the exp-Euler step-source form (task #16).
        # Only used when self.dt_correct_nmda is True.
        nmda_source_scale = 1.0 - math.exp(-self.dt / self.tau_syn)
        input_step_scale = 1.0 / float(self.n_substeps)
        istdp_decay = torch.exp(torch.tensor(-self.dt / self.tau_post_i,
                                             device=self.device, dtype=torch.float32))
        rate_alpha = self.dt / self.rate_avg_tau
        dt_s = self.dt / 1000.0
        spike_threshold = 30.0

        # ---- feedforward input ----
        I_A_input = self.input_scaling * (xA_batch @ self.W_inA + self.b_uniA)
        I_V_input = self.input_scaling * (xV_batch @ self.W_inV + self.b_uniV)

        # ---- local weight aliases (avoid repeated getattr) ----
        W_a2msi_AMPA = self.W_a2msi_AMPA
        W_v2msi_AMPA = self.W_v2msi_AMPA
        W_a2msi_NMDA = self.W_a2msi_NMDA
        W_v2msi_NMDA = self.W_v2msi_NMDA
        W_inA_inh = self.W_inA_inh
        W_inV_inh = self.W_inV_inh
        W_a2msiInh_AMPA = self.W_a2msiInh_AMPA
        W_a2msiInh_NMDA = self.W_a2msiInh_NMDA
        W_v2msiInh_AMPA = self.W_v2msiInh_AMPA
        W_v2msiInh_NMDA = self.W_v2msiInh_NMDA
        W_msiInh2Exc_GABA = self.W_msiInh2Exc_GABA
        W_msi2out = self.W_msi2out

        # ---- ring-buffer local aliases ----
        buf_a2msi = self.buffer_a2msi
        buf_v2msi = self.buffer_v2msi
        buf_inA_inh = self.buffer_inA_inh
        buf_inV_inh = self.buffer_inV_inh
        buf_a2msi_inh = self.buffer_a2msi_inh
        buf_v2msi_inh = self.buffer_v2msi_inh
        buf_msi_inh2exc = self.buffer_msi_inh2exc
        buf_msi2out = self.buffer_msi2out

        pos_a2msi = self._delay_positions["buffer_a2msi"]
        pos_v2msi = self._delay_positions["buffer_v2msi"]
        pos_inA_inh = self._delay_positions["buffer_inA_inh"]
        pos_inV_inh = self._delay_positions["buffer_inV_inh"]
        pos_a2msi_inh = self._delay_positions["buffer_a2msi_inh"]
        pos_v2msi_inh = self._delay_positions["buffer_v2msi_inh"]
        pos_msi_inh2exc = self._delay_positions["buffer_msi_inh2exc"]
        pos_msi2out = self._delay_positions["buffer_msi2out"]

        # task #27: when dt_correct_nmda is True, derive substep delays from
        # physical-ms attributes so the physical delay duration is dt-invariant.
        # Otherwise use the legacy substep-stored integer attributes.
        if self.dt_correct_nmda and hasattr(self, 'conduction_delay_a2msi_ms'):
            delay_a2msi       = self._delay_substeps_from_ms(self.conduction_delay_a2msi_ms)
            delay_v2msi       = self._delay_substeps_from_ms(self.conduction_delay_v2msi_ms)
            delay_inA_inh     = self._delay_substeps_from_ms(self.conduction_delay_inA_inh_ms)
            delay_inV_inh     = self._delay_substeps_from_ms(self.conduction_delay_inV_inh_ms)
            delay_a2msi_inh   = self._delay_substeps_from_ms(self.conduction_delay_a2msi_inh_ms)
            delay_v2msi_inh   = self._delay_substeps_from_ms(self.conduction_delay_v2msi_inh_ms)
            delay_msi_inh2exc = self._delay_substeps_from_ms(self.conduction_delay_msi_inh2exc_ms)
            delay_msi2out     = self._delay_substeps_from_ms(self.conduction_delay_msi2out_ms)
        else:
            delay_a2msi = self.conduction_delay_a2msi
            delay_v2msi = self.conduction_delay_v2msi
            delay_inA_inh = self.conduction_delay_inA_inh
            delay_inV_inh = self.conduction_delay_inV_inh
            delay_a2msi_inh = self.conduction_delay_a2msi_inh
            delay_v2msi_inh = self.conduction_delay_v2msi_inh
            delay_msi_inh2exc = self.conduction_delay_msi_inh2exc
            delay_msi2out = self.conduction_delay_msi2out

        zero_exc = torch.zeros((batch_size, self.n), device=self.device)
        zero_inh = torch.zeros((batch_size, self.n_inh), device=self.device)

        # ---- accumulate debug counters on GPU, sync once at end ----
        dbg_spk_A = torch.tensor(0.0, device=self.device)
        dbg_spk_V = torch.tensor(0.0, device=self.device)
        dbg_spk_M = torch.tensor(0.0, device=self.device)

        for sub_i in range(self.n_substeps):
            # --- Decay old currents ---
            self.I_A.mul_(decay_factor)
            self.I_V.mul_(decay_factor)
            self.I_M.mul_(decay_factor)
            self.I_M_inh.mul_(decay_factor)
            self.I_O.mul_(decay_factor)

            # Add external input (split across substeps)
            self.I_A.add_(I_A_input * input_step_scale)
            self.I_V.add_(I_V_input * input_step_scale)

            # --- Ring-buffer reads (replaces deque popleft) ---
            delayed_spikes_a2msi = buf_a2msi[pos_a2msi] if delay_a2msi > 0 else zero_exc
            delayed_spikes_v2msi = buf_v2msi[pos_v2msi] if delay_v2msi > 0 else zero_exc
            delayed_spikes_inA_inh = buf_inA_inh[pos_inA_inh] if delay_inA_inh > 0 else zero_exc
            delayed_spikes_inV_inh = buf_inV_inh[pos_inV_inh] if delay_inV_inh > 0 else zero_exc
            delayed_spikes_a2msi_inh = buf_a2msi_inh[pos_a2msi_inh] if delay_a2msi_inh > 0 else zero_exc
            delayed_spikes_v2msi_inh = buf_v2msi_inh[pos_v2msi_inh] if delay_v2msi_inh > 0 else zero_exc
            delayed_spikes_msi_inh2exc = buf_msi_inh2exc[pos_msi_inh2exc] if delay_msi_inh2exc > 0 else zero_inh
            delayed_spikes_msi2out = buf_msi2out[pos_msi2out] if delay_msi2out > 0 else zero_exc

            # ============== A->MSI (AMPA+NMDA) ==============
            # (Tsodyks-Markram STP usage for A->MSI)
            self.I_ampa_filtered.mul_(decay_factor)

            self.R_a += (1.0 - self.R_a) * (self.dt / self.tau_rec)
            use_A = self.u_a * self.R_a
            self.R_a -= use_A * delayed_spikes_a2msi
            I_M_a_AMPA = F.linear(use_A * delayed_spikes_a2msi, W_a2msi_AMPA)

            self.R_v += (1.0 - self.R_v) * (self.dt / self.tau_rec)
            use_V = self.u_v * self.R_v
            self.R_v -= use_V * delayed_spikes_v2msi
            I_M_v_AMPA = F.linear(use_V * delayed_spikes_v2msi, W_v2msi_AMPA)

            # -----------------------------------------------------------------
            # -----------------------------------------------------------------
            self.ampa_m.mul_(ampa_decay)
            self.ampa_m.add_(self.ampa_alpha *
                             (I_M_a_AMPA + I_M_v_AMPA).clamp(min=0.0))

            # -------------------------------------------------------------
            I_AMPA_curr = (self.gAMPA
                           * (I_M_a_AMPA + I_M_v_AMPA)
                           * (self.Erev_ampa - self.v_msi))
            # task #123 FIX 4: dt-invariant AMPA injection.
            # I_AMPA_curr is a spike-event-driven source added to the decaying
            # I_M state. Bare-add per substep is dt-dependent — at dt=0.05
            # I_AMPA_curr is added 2× more often, doubling the per-ms drive.
            # `dt_linear_scale = dt/0.1` (defined L1995) preserves total injection
            # per ms invariant. b_msi is the (zero-initialised) bias term and
            # has 0 measured contribution per debugger #122 audit (P02); left
            # bare-added for legibility.
            self.I_M.add_(I_AMPA_curr * dt_linear_scale + self.b_msi)

            I_ampa_lp = self.gAMPA_LP * self.ampa_m * (self.Erev_ampa - self.v_msi)

            self.I_ampa_filtered.add_(I_M_a_AMPA + I_M_v_AMPA)
            ampa_release = (I_M_a_AMPA + I_M_v_AMPA).clamp(min=0)  # (B, n)
            I_ampa_step = self.gAMPA * ampa_release * (self.Erev_ampa - self.v_msi)

            # -------------------------------------------------------
            # -------------------------------------------------------
            scale_F = 0.8  # 0 = no STD, 1 = same as AMPA
            gate_A_nmda = 1.0 - scale_F * (1.0 - self.R_a)  # (B,n)
            gate_V_nmda = 1.0 - scale_F * (1.0 - self.R_v)

            pre_A_nmda = gate_A_nmda * delayed_spikes_a2msi  # (B,n)
            pre_V_nmda = gate_V_nmda * delayed_spikes_v2msi
            # ---------------------------------------------------------------

            nmda_a = F.linear(pre_A_nmda, W_a2msi_NMDA)
            nmda_v = F.linear(pre_V_nmda, W_v2msi_NMDA)
            inc_m_exc = self.nmda_alpha * (nmda_a + nmda_v)  # unchanged
            self.nmda_m.mul_(nmda_decay)
            self.nmda_m.add_(inc_m_exc)

            # Dend coupling
            d_va = self.dend_coupling_alpha * (self.v_msi - self.v_dend_A) / self.tau_m
            d_vv = self.dend_coupling_alpha * (self.v_msi - self.v_dend_V) / self.tau_m
            self.v_dend_A += self.dt * d_va
            self.v_dend_V += self.dt * d_vv

            dv_nmda = ((self.v_msi + self.nmda_vrest_offset) - self.v_nmda) / self.tau_nmdaVolt
            self.v_nmda += self.dt * dv_nmda
            mg_A = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_A - self.mg_vhalf)))
            mg_V = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_V - self.mg_vhalf)))
            I_nmda = self.gNMDA * self.nmda_m * (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)
            I_nmda_step = self.gNMDA * inc_m_exc * (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)
            I_nmda_lp = self.gNMDA * self.nmda_m * (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)

            # task #16: dt-correct NMDA->I_M coupling (exp-Euler step-source form).
            # Flag OFF preserves legacy bare-add (dt-dependent); flag ON scales source
            # by (1 - exp(-dt/tau_syn)). See debug_dt/final_proof.py Fix B for proof.
            # task #147: reverted from task #135's `* dt_linear_scale` (×1.0 over-drive).
            if self.dt_correct_nmda:
                self.I_M.add_(I_nmda * nmda_source_scale)
            else:
                self.I_M.add_(I_nmda)

            release = (I_M_a_AMPA + I_M_v_AMPA)  # what you already had
            I_AMPA_tp = self.gAMPA * release * (self.Erev_ampa - self.v_msi)  # current
            J_ampa_step = I_AMPA_tp.detach()
            I_nmda_step = self.gNMDA * inc_m_exc * (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)
            I_ampa_total = I_AMPA_curr + self.gAMPA_LP * self.ampa_m * (self.Erev_ampa - self.v_msi)
            I_nmda_total = I_nmda  # already includes nmda_m tail

            if self.enable_probe and self._probe is not None:
                # --- 1. true instantaneous currents -------------------------
                J_ampa_step = I_AMPA_curr.detach()
                J_nmda_step = I_nmda_step.detach()  # B

                dt_sec = dt_s
                Q_ampa = I_ampa_total.detach() * dt_s
                Q_nmda = I_nmda_total.detach() * dt_s

                # ---- per‑spike injection (optional) ----
                J_ampa_inj = I_AMPA_curr.detach()
                J_nmda_inj = I_nmda_step.detach()

                # ---- spike counter for normalisation ----
                n_new_spk = (delayed_spikes_a2msi + delayed_spikes_v2msi).sum().item()

                # effective AMPA current
                # ------------------------------------------------------------------
                # ------------------------------------------------------------------
                release = (I_M_a_AMPA + I_M_v_AMPA)  # (B, n)

                # true AMPA current
                I_AMPA_step = self.gAMPA * release * (self.Erev_ampa - self.v_msi)
                J_ampa_inst = I_AMPA_curr.detach()  # <<<<<< line A
                J_nmda_inst = I_nmda_step.detach()  # unchanged
                # ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑#

                # (nothing else in this block changes)
                self._probe.log(
                    Q_ampa=Q_ampa,
                    Q_nmda=Q_nmda,
                    I_M=self.I_M.detach(),
                    sA=self._latest_sA, sV=self._latest_sV, sM=self._latest_sMSI,
                    R_a=self.R_a, R_v=self.R_v,
                    mg_gate=(mg_A + mg_V) / 2,
                    J_ampa_inst=J_ampa_inj,  # <<<<<< line B
                    J_nmda_inst=J_nmda_inj,
                    n_spikes=n_new_spk  # unchanged
                )

            I_inA_inh = F.linear(delayed_spikes_inA_inh, W_inA_inh)
            I_inV_inh = F.linear(delayed_spikes_inV_inh, W_inV_inh)
            # task #125: REVERTED task #123 FIX 2 (`* dt_linear_scale`). Debugger
            # #124 found the FF inh source-scaling breaks natural homeostatic
            # compensation — at dt=0.05, halving the FF inh per-substep drive
            # removes the very mechanism that compensates for the dt-invariant
            # excitatory side. Bare-add restored as the physically meaningful
            # interaction. AGC physical-time gating (FIX 1) and AMPA scaling
            # (FIX 4) are kept; NMDA Form 2 (FIX 5) is kept.
            self.I_M.sub_(self.g_FFinh
                          * (I_inA_inh + I_inV_inh))

            # ============== A->MSI_inh, V->MSI_inh ==============
            self.R_a_inh += (1.0 - self.R_a_inh) * (self.dt / self.tau_rec)
            use_A_inh = self.u_a_inh * self.R_a_inh
            spike_sum_a = delayed_spikes_a2msi_inh.sum(dim=1, keepdim=True)
            self.R_a_inh -= use_A_inh * spike_sum_a

            raw_inp_a_AMPA = F.linear(delayed_spikes_a2msi_inh, W_a2msiInh_AMPA)
            I_Mi_a_AMPA = (use_A_inh * raw_inp_a_AMPA)

            raw_inp_a_NMDA = F.linear(delayed_spikes_a2msi_inh, W_a2msiInh_NMDA)

            self.R_v_inh += (1.0 - self.R_v_inh) * (self.dt / self.tau_rec)
            use_V_inh = self.u_v_inh * self.R_v_inh
            spike_sum_v = delayed_spikes_v2msi_inh.sum(dim=1, keepdim=True)
            self.R_v_inh -= use_V_inh * spike_sum_v

            raw_inp_v_AMPA = F.linear(delayed_spikes_v2msi_inh, W_v2msiInh_AMPA)
            I_Mi_v_AMPA = (use_V_inh * raw_inp_v_AMPA)

            raw_inp_v_NMDA = F.linear(delayed_spikes_v2msi_inh, W_v2msiInh_NMDA)

            self.I_M_inh.add_(I_Mi_a_AMPA + I_Mi_v_AMPA + self.b_msi_inh)
            self.nmda_m_inh.mul_(nmda_decay)
            self.nmda_m_inh.add_(self.nmda_alpha * (raw_inp_a_NMDA + raw_inp_v_NMDA))

            d_viA = self.dend_coupling_alpha * (self.v_msi_inh - self.v_dend_inhA) / self.tau_m
            d_viV = self.dend_coupling_alpha * (self.v_msi_inh - self.v_dend_inhV) / self.tau_m
            self.v_dend_inhA += self.dt * d_viA
            self.v_dend_inhV += self.dt * d_viV

            dv_nmda_inh = ((self.v_msi_inh + self.nmda_vrest_offset) - self.v_nmda_inh) / self.tau_nmdaVolt
            self.v_nmda_inh += self.dt * dv_nmda_inh
            mg_iA = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_inhA - self.mg_vhalf)))
            mg_iV = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_inhV - self.mg_vhalf)))
            I_nmda_inh = self.gNMDA * self.nmda_m_inh * (mg_iA + mg_iV) * (self.Erev_nmda - self.v_msi_inh)
            # task #16: dt-correct NMDA->I_M_inh coupling (exp-Euler step-source form).
            # Same tau_syn as excitatory path (no separate tau_syn_inh in this model).
            # task #147: reverted from task #135's `* dt_linear_scale` (×1.0 over-drive).
            if self.dt_correct_nmda:
                self.I_M_inh.add_(I_nmda_inh * nmda_source_scale)
            else:
                self.I_M_inh.add_(I_nmda_inh)

            # MSI_inh->MSI_ex
            I_M_inh2exc = F.linear(delayed_spikes_msi_inh2exc, W_msiInh2Exc_GABA)
            # task #125: REVERTED task #123 FIX 3 (`* dt_linear_scale`). Same
            # rationale as FIX 2 revert above — debugger #124 found GABA recurrent
            # source-scaling also breaks the natural homeostatic balance.
            self.I_M.sub_(I_M_inh2exc)

            # MSI->Out
            I_O_msi = F.linear(delayed_spikes_msi2out, W_msi2out)
            self.I_O.add_((I_O_msi + self.b_out))

            I_latA = torch.mm(self._latest_sA, self.W_latA)  # shape (B, n)
            self.I_A.sub_(self.g_latA * I_latA)

            I_latV = torch.mm(self._latest_sV, self.W_latV)  # shape (B, n)
            self.I_V.sub_(self.g_latV * I_latV)

            # -------------- Izhikevich updates --------------
            # A
            dVA = (0.04 * self.v_uniA.pow(2) + 5.0 * self.v_uniA + 140.0
                   - self.u_uniA + self.I_A)
            self.v_uniA += self.dt * dVA
            self.u_uniA += self.dt * (self.aA * (self.bA * self.v_uniA - self.u_uniA))
            spike_mask_A = (self.v_uniA >= spike_threshold)
            new_sA = spike_mask_A.float()
            self.v_uniA.masked_fill_(spike_mask_A, self.cA)
            self.u_uniA[spike_mask_A] += self.dA

            # V
            dVV = (0.04 * self.v_uniV.pow(2) + 5.0 * self.v_uniV + 140.0
                   - self.u_uniV + self.I_V)
            self.v_uniV += self.dt * dVV
            self.u_uniV += self.dt * (self.aV * (self.bV * self.v_uniV - self.u_uniV))
            spike_mask_V = (self.v_uniV >= spike_threshold)
            new_sV = spike_mask_V.float()
            self.v_uniV.masked_fill_(spike_mask_V, self.cV)
            self.u_uniV[spike_mask_V] += self.dV

            # MSI excit
            dVM = (0.04 * self.v_msi.pow(2) + 5.0 * self.v_msi + 140.0 - self.u_msi + self.I_M)
            self.v_msi += self.dt * dVM
            self.u_msi += self.dt * (self.aM * (self.bM * self.v_msi - self.u_msi))
            spike_mask_M = (self.v_msi >= spike_threshold)
            new_sM = spike_mask_M.float()
            self.v_msi.masked_fill_(spike_mask_M, self.cM)
            self.u_msi[spike_mask_M] += self.dM

            # (A) compute surround inhibition current
            I_latM = torch.mm(new_sM, self.W_MSI_inh)  # shape (B, n)
            # (B) apply it
            self.I_M.sub_(self.g_GABA * I_latM)

            # ── E/I component recording (separate synaptic currents) ──
            if self._ei_record is not None:
                # Excitatory onto MSI excitatory
                _I_E_ampa = torch.clamp(I_AMPA_curr, min=0.0)
                # Task #42 fix: mirror the actual NMDA->I_M injection site.
                # When dt_correct_nmda is True, NMDA is injected as
                # I_nmda * nmda_source_scale (exp-Euler step-source form),
                # so the probe must record the same scaled current to stay symmetric.
                # task #147: reverted from task #135's `* dt_linear_scale` mirror.
                if self.dt_correct_nmda:
                    _I_E_nmda = torch.clamp(I_nmda * nmda_source_scale, min=0.0)
                else:
                    _I_E_nmda = torch.clamp(I_nmda, min=0.0)
                _I_E = _I_E_ampa + _I_E_nmda
                # Inhibitory onto MSI excitatory
                _I_I_ff = torch.clamp(self.g_FFinh * (I_inA_inh + I_inV_inh), min=0.0)
                _I_I_recur = torch.clamp(I_M_inh2exc, min=0.0)
                _I_I_lat = torch.clamp(self.g_GABA * I_latM, min=0.0)
                _I_I = _I_I_ff + _I_I_recur + _I_I_lat

                self._ei_record["I_E_mean"].append(_I_E.mean().item())
                self._ei_record["I_I_mean"].append(_I_I.mean().item())
                self._ei_record["Q_E"].append((_I_E * dt_s).mean().item())
                self._ei_record["Q_I"].append((_I_I * dt_s).mean().item())
                self._ei_record["AMPA"].append(_I_E_ampa.mean().item())
                self._ei_record["NMDA"].append(_I_E_nmda.mean().item())
                self._ei_record["FFInh"].append(_I_I_ff.mean().item())
                self._ei_record["RecurInh"].append(_I_I_recur.mean().item())
                self._ei_record["LatInh"].append(_I_I_lat.mean().item())

            if self.enable_probe and self._probe is not None:
                I_total = self.I_M.detach()  # includes inhibition
                Q_exc = torch.clamp(I_total, min=0) * dt_s
                Q_inh = -torch.clamp(I_total, max=0) * dt_s
                self._probe.log_EI(Q_exc, Q_inh)  # add two extra slots

            if epoch_idx == 5:
                # instantaneous excitation
                exc_AMPA = ((I_M_a_AMPA + I_M_v_AMPA).clamp(min=0) *
                            self.gAMPA * (self.Erev_ampa - self.v_msi)).sum().item()
                exc_NMDA = (self.gNMDA * inc_m_exc *
                            (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)).sum().item()

                # instantaneous inhibition
                inh_FF = (self.g_FFinh * (I_inA_inh + I_inV_inh)).sum().item()
                inh_lat = (self.g_GABA * I_latM).sum().item()
                inh_recur = I_M_inh2exc.sum().item()


            if return_spike_sum:  # ****
                sum_sM += new_sM  # ****

            # iSTDP trace (decay pre-computed outside loop)
            self.post_i_trace.mul_(istdp_decay)
            self.post_i_trace.add_(new_sM)

            self.post_rate_avg.mul_(1.0 - rate_alpha).add_(rate_alpha * new_sM)

            # MSI inh
            dVMi = (0.04 * self.v_msi_inh.pow(2) + 5.0 * self.v_msi_inh + 140.0 - self.u_msi_inh + self.I_M_inh)
            self.v_msi_inh += self.dt * dVMi
            self.u_msi_inh += self.dt * (self.aMi * (self.bMi * self.v_msi_inh - self.u_msi_inh))
            spike_mask_Mi = (self.v_msi_inh >= spike_threshold)
            new_sMi = spike_mask_Mi.float()
            self.v_msi_inh.masked_fill_(spike_mask_Mi, self.cMi)
            self.u_msi_inh[spike_mask_Mi] += self.dMi

            # Out
            dVO = (0.04 * self.v_out.pow(2) + 5.0 * self.v_out + 140.0 - self.u_out + self.I_O)
            self.v_out += self.dt * dVO
            self.u_out += self.dt * (self.aO * (self.bO * self.v_out - self.u_out))
            spike_mask_O = (self.v_out >= spike_threshold)
            new_sO = spike_mask_O.float()
            self.v_out.masked_fill_(spike_mask_O, self.cO)
            self.u_out[spike_mask_O] += self.dO

            # iSTDP homeostasis on MSI_inh->MSI_ex if allowed
            if self.allow_inhib_plasticity and getattr(self, 'plasticity_enabled', True):
                pre_ff_inh = torch.cat([delayed_spikes_inA_inh,
                                        delayed_spikes_inV_inh], dim=1)  # (B, 2n)

                W_ff_inh = torch.cat([self.W_inA_inh, self.W_inV_inh], dim=1)

                # Vogel‑Abbott update
                dw = self.eta_i * torch.bmm(
                    (self.post_i_trace - self.rho0).unsqueeze(2),
                    pre_ff_inh.unsqueeze(1)  # (B ,1 ,2n)
                ).mean(0)

                dw_A = dw[:, :self.n]
                dw_V = dw[:, self.n:]
                self._p_add("W_inA_inh", dw_A)
                self._p_add("W_inV_inh", dw_V)

            if valid_mask is not None:
                mask_sub = valid_mask.view(-1, 1)
                new_sA *= mask_sub
                new_sV *= mask_sub
                new_sM *= mask_sub
                new_sMi *= mask_sub
                new_sO *= mask_sub

            # Ring-buffer writes (replaces deque append)
            if delay_a2msi > 0:
                buf_a2msi[pos_a2msi].copy_(new_sA)
                pos_a2msi = (pos_a2msi + 1) % delay_a2msi
            if delay_v2msi > 0:
                buf_v2msi[pos_v2msi].copy_(new_sV)
                pos_v2msi = (pos_v2msi + 1) % delay_v2msi
            if delay_inA_inh > 0:
                buf_inA_inh[pos_inA_inh].copy_(new_sA)
                pos_inA_inh = (pos_inA_inh + 1) % delay_inA_inh
            if delay_inV_inh > 0:
                buf_inV_inh[pos_inV_inh].copy_(new_sV)
                pos_inV_inh = (pos_inV_inh + 1) % delay_inV_inh
            if delay_a2msi_inh > 0:
                buf_a2msi_inh[pos_a2msi_inh].copy_(new_sA)
                pos_a2msi_inh = (pos_a2msi_inh + 1) % delay_a2msi_inh
            if delay_v2msi_inh > 0:
                buf_v2msi_inh[pos_v2msi_inh].copy_(new_sV)
                pos_v2msi_inh = (pos_v2msi_inh + 1) % delay_v2msi_inh
            if delay_msi_inh2exc > 0:
                buf_msi_inh2exc[pos_msi_inh2exc].copy_(new_sMi)
                pos_msi_inh2exc = (pos_msi_inh2exc + 1) % delay_msi_inh2exc
            if delay_msi2out > 0:
                buf_msi2out[pos_msi2out].copy_(new_sM)
                pos_msi2out = (pos_msi2out + 1) % delay_msi2out

            sA, sV, sM, sMi, sO = new_sA, new_sV, new_sM, new_sMi, new_sO

            self._latest_sA = sA
            self._latest_sV = sV
            self._latest_sMSI = sM
            self._latest_sMSI_inh = sMi

            # ---------- epoch-level debug counters (GPU accumulate) ----
            dbg_spk_A += sA.sum()
            dbg_spk_V += sV.sum()
            dbg_spk_M += sM.sum()

            if curr_debug:
                self.debug_msi_current_and_stp(sub_i)

            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            if sub_i % 10 == 0 and getattr(self, 'plasticity_enabled', True):
                apply_topographic_anchor_unimodal(self, layer="A",
                                                  lr=0.1 * self.lr_uni,
                                                  sigma=3.0)
                apply_topographic_anchor_unimodal(self, layer="V",
                                                  lr=0.1 * self.lr_uni,
                                                  sigma=3.0)
                if epoch_idx > 25:
                    apply_topographic_anchor_msi(self, layer="A",
                                                 lr=0.8 * self.lr_msi,
                                                 sigma=3.0)
                    apply_topographic_anchor_msi(self, layer="V",
                                                 lr=0.8 * self.lr_msi,
                                                 sigma=3.0)

            self.step_counter += 1

        # ------ write back ring-buffer positions ------
        self._delay_positions["buffer_a2msi"] = pos_a2msi
        self._delay_positions["buffer_v2msi"] = pos_v2msi
        self._delay_positions["buffer_inA_inh"] = pos_inA_inh
        self._delay_positions["buffer_inV_inh"] = pos_inV_inh
        self._delay_positions["buffer_a2msi_inh"] = pos_a2msi_inh
        self._delay_positions["buffer_v2msi_inh"] = pos_v2msi_inh
        self._delay_positions["buffer_msi_inh2exc"] = pos_msi_inh2exc
        self._delay_positions["buffer_msi2out"] = pos_msi2out

        # ------ sync debug counters (one GPU->CPU transfer) ------
        self._dbg_spk_A += dbg_spk_A.item()
        self._dbg_spk_V += dbg_spk_V.item()
        self._dbg_spk_MSI += dbg_spk_M.item()
        self._dbg_steps += batch_size

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        if getattr(self, 'plasticity_enabled', True):
            apply_topographic_anchor_unimodal(self, layer="A",
                                              lr=1.0 * self.lr_uni,  # was 0.5
                                              sigma=2.5)  # was 3.0
            apply_topographic_anchor_unimodal(self, layer="V",
                                              lr=1.0 * self.lr_uni,
                                              sigma=2.5)

            apply_local_competition_unimodal_fast(self, "A",
                                                  beta=2.0 * self.lr_uni,  # was 0.8
                                                  neighbour_dist=4)
            apply_local_competition_unimodal_fast(self, "V",
                                                  beta=2.0 * self.lr_uni,
                                                  neighbour_dist=4)

            if epoch_idx > 25:
                apply_local_competition_msi_fast(self,
                                                 beta=1.5 * self.lr_msi,
                                                 neighbour_dist=6)

            soft_row_scaling(self)  # keeps norms near unity but *does not* freeze patterns

            # very slow scaling
            # task #131: REVERTED task #107 Phase B ramp. All 4 MSI-input weight
            # matrices now use the default target_mean=0.006 — the same value
            # the W_inA / W_inV (Oja-coregulated) matrices use. The Phase B
            # log-linear ramp to 0.00012 caused the W_a2msi_* weights to shrink
            # 10-26× below legacy and silenced the MSI population by ep ~50
            # (debugger #130 root cause for the task #128 retrain collapse).
            if (self.step_counter % 10) == 0:
                slow_synaptic_scaling(self.W_inA)         # default 0.006
                slow_synaptic_scaling(self.W_inV)         # default 0.006
                slow_synaptic_scaling(self.W_a2msi_AMPA, target_mean=0.006)
                slow_synaptic_scaling(self.W_v2msi_AMPA, target_mean=0.006)
                slow_synaptic_scaling(self.W_a2msi_NMDA, target_mean=0.006)
                slow_synaptic_scaling(self.W_v2msi_NMDA, target_mean=0.006)

        if not getattr(self, 'freeze_g_FFinh', False):
            # task #123 FIX 1: physical-time-gated AGC cadence (was
            # `step_counter %% N`). AGC homeostatic loops should fire at
            # physical (ms) cadence, not substep-count cadence. Preserves
            # dt=0.1 firing pattern exactly: fast every 0.1 ms = every substep;
            # slow every 10 ms = every 100 substeps. Debugger #122 ranked AGC
            # cadence as the single largest dt-sensitivity contributor
            # (+40.6 ms drift; AGC-frozen vs AGC-on isolation probe).
            current_t_ms = self.step_counter * self.dt

            # --- Fast AGC (PV-like) -------------------------------------------
            if (current_t_ms - self._last_agc_fast_t) >= self.T_AGC_FAST_MS:
                self._last_agc_fast_t = current_t_ms
                exc_fast = self.I_ampa_filtered.clamp(min=0).mean().item()  # AMPA only
                inh_mean = (-self.I_M).clamp(min=0).mean().item()
                target_ratio = self.targ_ratio
                alpha_fast = 1e-3
                self.g_FFinh += alpha_fast * (exc_fast * target_ratio - inh_mean)
                self.g_FFinh = max(0.05, min(self.g_FFinh, 5.0))  # task #128: REVERTED task #80 INT-4

            # --- Slow AGC (homeostatic; physical 10 ms cadence) ---------------
            if (current_t_ms - self._last_agc_slow_t) >= self.T_AGC_SLOW_MS:
                self._last_agc_slow_t = current_t_ms
                exc_mean_long = self.I_M.clamp(min=0).mean().item()
                inh_mean_long = (-self.I_M).clamp(min=0).mean().item()
                alpha_slow = 2e-4
                self.g_FFinh += alpha_slow * (exc_mean_long * self.pv_nmda - inh_mean_long)
                self.g_FFinh = max(0.05, min(self.g_FFinh, 5.0))  # task #128: REVERTED task #80 INT-4

        if return_delayed and return_spike_sum:
            return (sA, sV, sM, sO,
                    delayed_spikes_a2msi, delayed_spikes_v2msi,
                    sum_sM)  # 7 objs
        elif return_delayed:  # delayed spikes
            return (sA, sV, sM, sO,
                    delayed_spikes_a2msi, delayed_spikes_v2msi)  # 6 objs
        elif return_spike_sum:  # only spike accumulator
            return sA, sV, sM, sO, sum_sM  # 5 objs
        else:  # vanilla
            return sA, sV, sM, sO  # 4 objs

    # ─────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────
    def stdp_update_batch(self,
                          W_attr: str,
                          post_spk: torch.Tensor,
                          pre_spk: torch.Tensor,
                          post_trace: torch.Tensor,
                          pre_trace: torch.Tensor,
                          lr: float,
                          tau_pre: float = 0.9,
                          tau_post: float = 0.9,
                          A_plus: float = 1.0,
                          A_minus: float = 1.0,
                          debug=False):
        """
        Pair-based STDP update. Now with debug logs.
        """
        B = post_spk.size(0)

        # Update the eligibility traces
        pre_trace.mul_(tau_pre).add_(pre_spk)
        post_trace.mul_(tau_post).add_(post_spk)

        dW = torch.zeros_like(getattr(self, W_attr))
        for b in range(B):
            dW += A_plus * torch.ger(post_spk[b], pre_trace[b]) \
                  - A_minus * torch.ger(post_trace[b], pre_spk[b])

        dW.mul_(lr / B)

        # Apply via the positive reparam
        self._p_add(W_attr, dW)

        return pre_trace, post_trace, dW

    def normalize_rows_gpu(self, W):
        norms = torch.norm(W, dim=1, keepdim=True)
        norms[norms == 0] = 1.0
        W.div_(norms)

    def normalize_rows(self, attr: str, eps: float = 1e-8):
        with torch.no_grad():
            W = getattr(self, attr)  # current positive view
            theta = self.parametrizations[attr].original
            P = self.parametrizations[attr][0]

            Wnorm = W / (W.norm(2, 1, keepdim=True) + eps)
            theta.copy_(P.right_inverse(Wnorm))

    ########################################################
    #           UNSUPERVISED & SUPERVISED TRAINING
    ########################################################

    def sample_poisson_spikes_from_analog(self, analog_vec, max_rate=50.0, dt=0.001):
        """
        Convert an analog input (shape (B,n) or (n,)) into a 0/1 spike train
        by sampling from Poisson( lambda = analog_vec * max_rate ),
        each step scaled by dt.

        If analog_vec is (n,), we treat it as (1,n).
        """
        if analog_vec.dim() == 1:
            analog_vec = analog_vec.unsqueeze(0)  # (1, n)

        rate = analog_vec.clamp(min=0) * max_rate * dt  # shape (B, n)
        p = rate.clamp(max=1.0)
        spikes = torch.bernoulli(p)
        return spikes  # same shape (B, n)

    def train_unsupervised_batch(self,
                                 n_sequences,
                                 batch_size: int = 32,
                                 debug: bool = False,
                                 epoch_idx: int = 0):
        """
        Unsupervised STDP phase:
        • Generates random AV event sequences
        • Runs the network
        • Applies STDP
            – In  → Uni  : uses Poisson-sampled presyn spikes (same as before)
            – Uni → MSI
        """

        seq_counter = 0  # how many sequences processed so far
        while seq_counter < n_sequences:

            B = min(batch_size,  # current mini-batch
                    n_sequences - seq_counter)

            # --- 1. generate synthetic sequences ------------------------------
            loc_seqs, mod_seqs, offset_ok, seq_lens = generate_event_loc_seq_batch(
                batch_size=B,
                space_size=self.space_size,
                offset_probability=0.6,
                temporal_jitter_max=2
            )

            T_max = max(seq_lens)

            batch_intensity = 10 ** torch.empty(1).uniform_(math.log10(0.4), 0).item()

            xA, xV, valid = generate_av_batch_tensor(
                loc_seqs, mod_seqs, offset_ok,
                n=self.n,
                space_size=self.space_size,
                sigma_in=self.sigma_in,
                noise_std=self.noise_std,
                loc_jitter_std=self.loc_jitter_std,
                stimulus_intensity=batch_intensity,
                device=self.device,
                max_len=T_max)

            self.reset_state(B)

            for t in range(T_max):
                # forward pass
                (sA, sV, sMSI, _,
                 dA2M, dV2M) = self.update_all_layers_batch(
                    xA[:, t],  # analog A input
                    xV[:, t],  # analog V input
                    valid[:, t],  # validity mask
                    epoch_idx=epoch_idx,
                    return_delayed=True)

                pre_inA = self.sample_poisson_spikes_from_analog(
                    xA[:, t], max_rate=300., dt=0.01)
                pre_inV = self.sample_poisson_spikes_from_analog(
                    xV[:, t], max_rate=300., dt=0.01)

                self.pre_trace_inA, self.post_trace_inA, _ = self.stdp_update_batch(
                    'W_inA',
                    post_spk=sA,
                    pre_spk=pre_inA,
                    post_trace=self.post_trace_inA,
                    pre_trace=self.pre_trace_inA,
                    lr=self.lr_uni,
                    debug=debug)

                self.pre_trace_inV, self.post_trace_inV, _ = self.stdp_update_batch(
                    'W_inV',
                    post_spk=sV,
                    pre_spk=pre_inV,
                    post_trace=self.post_trace_inV,
                    pre_trace=self.pre_trace_inV,
                    lr=self.lr_uni,
                    debug=debug)

                # Uni → MSI STDP
                self.pre_trace_a2msi, self.post_trace_a2msi, _ = self.stdp_update_batch(
                    'W_a2msi_AMPA',
                    post_spk=sMSI,
                    pre_spk=dA2M,  
                    post_trace=self.post_trace_a2msi,
                    pre_trace=self.pre_trace_a2msi,
                    lr=self.lr_msi,
                    debug=debug)

                self.pre_trace_v2msi, self.post_trace_v2msi, _ = self.stdp_update_batch(
                    'W_v2msi_AMPA',
                    post_spk=sMSI,
                    pre_spk=dV2M,  
                    post_trace=self.post_trace_v2msi,
                    pre_trace=self.pre_trace_v2msi,
                    lr=self.lr_msi,
                    debug=debug)

                # NMDA STDP (slower)

                nmda_lr_scale = 1  # NMDA learns slower
                self.pre_trace_a2msi_nmda, self.post_trace_a2msi_nmda, _ = self.stdp_update_batch(
                    'W_a2msi_NMDA',
                    post_spk=sMSI,
                    pre_spk=dA2M,
                    post_trace=self.post_trace_a2msi_nmda,
                    pre_trace=self.pre_trace_a2msi_nmda,
                    lr=self.lr_msi * nmda_lr_scale,
                    tau_pre=0.95,
                    tau_post=0.95,
                    A_plus=1,
                    A_minus=1,
                    debug=debug)

                self.pre_trace_v2msi_nmda, self.post_trace_v2msi_nmda, _ = self.stdp_update_batch(
                    'W_v2msi_NMDA',
                    post_spk=sMSI,
                    pre_spk=dV2M,
                    post_trace=self.post_trace_v2msi_nmda,
                    pre_trace=self.pre_trace_v2msi_nmda,
                    lr=self.lr_msi * nmda_lr_scale,
                    tau_pre=0.95,
                    tau_post=0.95,
                    A_plus=1,
                    A_minus=1,
                    debug=debug)

                # --- OPTIONAL: re-normalise AMPA/NMDA split -------------------
                # Task #66 fix: migrated from set_param_weight()/parametrizations
                # to direct .copy_() ops with _p_add-style clamp bounds
                # (eps=1e-9, abs_cap=5.0). The parametrize wrappers were removed
                # in commit 1644af6 (de-parametrize refactor); this site was
                # missed during that migration. See debug_dt/DIAGNOSTIC_REPORT_task65.md.
                with torch.no_grad():
                    # A -> MSI connections — redistribute 25/75 AMPA/NMDA
                    W_tot_a = self.W_a2msi_AMPA + self.W_a2msi_NMDA
                    self.W_a2msi_AMPA.copy_((0.25 * W_tot_a).clamp_(min=1e-9, max=5.0))
                    self.W_a2msi_NMDA.copy_((0.75 * W_tot_a).clamp_(min=1e-9, max=5.0))

                    # V -> MSI connections — redistribute 25/75 AMPA/NMDA
                    W_tot_v = self.W_v2msi_AMPA + self.W_v2msi_NMDA
                    self.W_v2msi_AMPA.copy_((0.25 * W_tot_v).clamp_(min=1e-9, max=5.0))
                    self.W_v2msi_NMDA.copy_((0.75 * W_tot_v).clamp_(min=1e-9, max=5.0))

            # end-for t
            seq_counter += B

            # (Optional) print diagnostics once per mini-batch
            if debug:
                print(f"[unsup] processed {seq_counter}/{n_sequences} sequences")

        # ----------------------------------------------------------------------
        # apply slow updates
        # ----------------------------------------------------------------------
        apply_topographic_anchor_unimodal(self, layer="A", lr=self.lr_uni)
        apply_topographic_anchor_unimodal(self, layer="V", lr=self.lr_uni)
        soft_row_scaling(self)

    def evaluate_batch(self,
                       n_sequences: int,
                       condition: str = "both",  # "both", "audio_only", "visual_only"
                       batch_size: int = 32,
                       stimulus_intensity: float = 1.0,
                       decode: str = "argmax"  # "com" (centre-of-mass) or "argmax"
                       ) -> float:
        """
        Returns the mean absolute localisation error (degrees) for *n_sequences*
        synthetic AV trials under *condition*.  Each trial may contain several
        events, but **only the last event in the sequence is evaluated**.

        Parameters
        ----------
        decode : {"com", "argmax"}
            • "com"    – centre-of-mass decoder (robust, default)
            • "argmax" – winner-take-all decoder
        """
        errors: list[float] = []

        while n_sequences > 0:
            B = min(batch_size, n_sequences)

            loc_seqs, mod_seqs, offset, seq_lens = generate_event_loc_seq_batch(
                batch_size=B,
                space_size=self.space_size,
                offset_probability=0.1
            )
            T_max = max(seq_lens)

            xA, xV, valid = generate_av_batch_tensor(
                loc_seqs, mod_seqs, offset,
                n=self.n,
                space_size=self.space_size,
                sigma_in=self.sigma_in,
                noise_std=self.noise_std,
                loc_jitter_std=self.loc_jitter_std,
                stimulus_intensity=stimulus_intensity,
                device=self.device,
                max_len=T_max
            )

            # optionally zero one modality
            if condition == "audio_only":
                xV.zero_()
            elif condition == "visual_only":
                xA.zero_()

            # --- 2. run the network -------------------------------------------
            self.reset_state(B)
            msi_hist = torch.zeros(T_max, B, self.n, device=self.device)

            for t in range(T_max):
                self.update_all_layers_batch(xA[:, t], xV[:, t], valid[:, t])
                msi_hist[t] = self._latest_sMSI

            for seq_i in range(B):
                loc_seq = loc_seqs[seq_i]

                # frames that actually carry a stimulus
                non_blank = [t for t, val in enumerate(loc_seq) if val != 999]
                if not non_blank:
                    continue  # this sequence had no events

                final_t = non_blank[-1] + 5  # last frame of last event

                first_t = final_t
                while first_t > 0 and loc_seq[first_t - 1] != 999:
                    first_t -= 1

                event_slice = slice(first_t, final_t + 1)
                spikes = msi_hist[event_slice, seq_i].sum(dim=0)

                # decode
                pred_deg = decode_msi_location(
                    spikes.unsqueeze(0),  # (1, n)
                    space_size=self.space_size,
                    method=decode
                )[0].item()

                true_deg = loc_seq[final_t]
                err = abs(pred_deg - true_deg)
                if err > 90:  # shortest angular distance
                    err = 180 - err
                errors.append(err)

            n_sequences -= B

        return 0.0 if not errors else float(np.mean(errors))

    # ---------------------------------------------------------------------------
    def evaluate_batch_argmax_10step(self,
                                     n_sequences: int,
                                     batch_size: int = 32,
                                     condition: str = "both") -> float:
        """
        Mean |error| (degrees) using
          • 10-frame stimulus pulse,
          • arg-max over total MSI spikes in those frames,
          • all spikes generated in each external frame.

        condition ∈ {"both", "audio_only", "visual_only"}
        """
        errors = []
        while n_sequences > 0:
            B = min(batch_size, n_sequences)

            # ---- build stimuli -------------------------------------------------
            loc_seqs, mod_seqs, offs, lens = generate_event_loc_seq_batch(
                batch_size=B,
                space_size=self.space_size,
                event_duration=10)  # <-- keep pulse length in sync
            T_max = max(lens)

            xA, xV, valid = generate_av_batch_tensor(
                loc_seqs, mod_seqs, offs,
                n=self.n,
                space_size=self.space_size,
                sigma_in=self.sigma_in,
                noise_std=self.noise_std,
                loc_jitter_std=self.loc_jitter_std,
                stimulus_intensity=1.0,
                device=self.device,
                max_len=T_max)

            if condition == "audio_only":
                xV.zero_()
            elif condition == "visual_only":
                xA.zero_()

            # ---- run network ---------------------------------------------------
            self.reset_state(B)
            msi_sum = torch.zeros(T_max, B, self.n, device=self.device)

            for t in range(T_max):
                *_, sSum = self.update_all_layers_batch(
                    xA[:, t], xV[:, t], valid[:, t],
                    return_spike_sum=True)
                msi_sum[t] = sSum

            # ---- decode each sequence -----------------------------------------
            for i in range(B):
                loc_seq = loc_seqs[i]
                pulse_frames = [t for t, v in enumerate(loc_seq) if v != 999]
                if not pulse_frames:
                    continue
                end = pulse_frames[-1]
                start = max(end - 9, 0)  # 10 frames

                spikes = msi_sum[start:end + 1, i].sum(0)
                pred_i = torch.argmax(spikes).item()
                pred_deg = index_to_location(pred_i, self.n, self.space_size)

                true_deg = loc_seq[end]
                err = abs(pred_deg - true_deg)
                if err > 90:  # shortest path on 0–180° circle
                    err = 180 - err
                errors.append(err)

            n_sequences -= B

        return 0.0 if not errors else float(np.mean(errors))

    # ─────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────
    def isolate_surround(self, enable: bool = True):
        """
        If *enable* is True, this zeroes out:
           • feed-forward A→MSI_ex & V→MSI_ex inhibition
           • MSI_inh → MSI_ex gate
        and freezes inhibitory plasticity, leaving the Mexican-hat
        surround (W_MSI_inh + g_GABA) untouched.
        Call again with False to restore learning.
        """
        z = 0.0 if enable else 1.0
        self.W_inA_inh.mul_(z)
        self.W_inV_inh.mul_(z)
        self.W_msiInh2Exc_GABA.mul_(z)
        self.allow_inhib_plasticity = not enable


def make_checkpoint(net,
                    epoch: int,
                    optim=None,  # pass your optimiser if you need it
                    comment="",
                    rng_tag=True):
    """
    Collect *all* state required to restore or resume the experiment.

    Parameters
    ----------
    net      : trained MultiBatchAudVisMSINetworkTime
    epoch    : int, last finished epoch  (for bookkeeping)
    optim    : torch.optim.Optimizer | None
               If provided, its state_dict is saved so you can resume training.
    comment  : str, optional text note
    rng_tag  : bool, also save torch RNG states (recommended)

    Returns
    -------
    checkpoint : dict  (ready to torch.save)
    """

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    constructor_hparams = dict(
        n_neurons=net.n,
        batch_size=net.batch_size,
        lr_unimodal=net.lr_uni,
        lr_msi=net.lr_msi,
        lr_readout=net.lr_out,
        sigma_in=net.sigma_in,
        noise_std=net.noise_std,
        v_thresh=net.v_thresh,
        dt=net.dt,
        tau_m=net.tau_m,
        n_substeps=net.n_substeps,
        loc_jitter_std=net.loc_jitter_std,
        space_size=net.space_size,
        conduction_delay_a2msi=net.conduction_delay_a2msi,
        conduction_delay_v2msi=net.conduction_delay_v2msi,
        conduction_delay_msi2out=net.conduction_delay_msi2out,
    )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    mutable_hparams = dict(
        # learning hyper-parameters / curriculum
        sigma_teacher_init=net.sigma_teacher_init,
        sigma_teacher_final=net.sigma_teacher_final,
        curriculum_epochs=net.curriculum_epochs,

        # global gains & scaling
        input_scaling=net.input_scaling,
        g_latA=net.g_latA,
        g_latV=net.g_latV,
        g_GABA=net.g_GABA,
        g_FFinh=net.g_FFinh,

        # MSI surround-inhibition hyper-params
        R_near=net.R_near,
        eta_H=net.eta_H,
        eta_AH=net.eta_AH,

        # synaptic-current / receptor time-constants
        gNMDA=net.gNMDA,
        tau_syn=net.tau_syn,
        tau_nmda=net.tau_nmda,
        nmda_alpha=net.nmda_alpha,
        mg_k=net.mg_k,
        Erev_nmda=net.Erev_nmda,
        tau_nmdaVolt=net.tau_nmdaVolt,
        v_nmda_rest=net.v_nmda_rest,
        nmda_vrest_offset=net.nmda_vrest_offset,
        mg_vhalf=net.mg_vhalf,
        dend_coupling_alpha=net.dend_coupling_alpha,

        # Tsodyks–Markram STP
        tau_rec=net.tau_rec,
        tau_fac=net.tau_fac,

        # iSTDP homeostasis
        rho0=net.rho0,
        eta_i=net.eta_i,
        tau_post_i=net.tau_post_i,
        rate_avg_tau=net.rate_avg_tau,

        # conduction delays outside the constructor
        conduction_delay_inA_inh=net.conduction_delay_inA_inh,
        conduction_delay_inV_inh=net.conduction_delay_inV_inh,
        conduction_delay_a2msi_inh=net.conduction_delay_a2msi_inh,
        conduction_delay_v2msi_inh=net.conduction_delay_v2msi_inh,
        conduction_delay_msi_inh2exc=net.conduction_delay_msi_inh2exc,

        # Izhikevich intrinsic parameters
        aA=net.aA, bA=net.bA, cA=net.cA, dA=net.dA,
        aV=net.aV, bV=net.bV, cV=net.cV, dV=net.dV,
        aM=net.aM, bM=net.bM, cM=net.cM, dM=net.dM,
        aMi=net.aMi, bMi=net.bMi, cMi=net.cMi, dMi=net.dMi,
        aO=net.aO, bO=net.bO, cO=net.cO, dO=net.dO,

        # plasticity & toggles
        allow_inhib_plasticity=net.allow_inhib_plasticity,

        # diagnostic counters
        step_counter=net.step_counter,
    )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    rng_state = dict()
    if rng_tag:
        rng_state["torch_cpu"] = torch.get_rng_state()
        rng_state["torch_cuda"] = (torch.cuda.get_rng_state()
                                   if torch.cuda.is_available() else None)

    # ------------------------------------------------------------------
    # D. pack everything together
    # ------------------------------------------------------------------
    checkpoint = dict(
        model_state=net.state_dict(),  # *all* Parameters + buffers
        constructor_hparams=constructor_hparams,
        mutable_hparams=mutable_hparams,
        epoch=epoch,
        comment=comment,
        **({"optim_state": optim.state_dict()} if optim else {}),
        **rng_state,
        timestamp=datetime.utcnow().isoformat(timespec="seconds")
    )
    return checkpoint


# Diagnostics functions
import math
import torch
from collections import defaultdict
from typing import Literal


# Task #68: removed @torch.inference_mode() decorator. inference_mode marks
# tensors created inside as "inference tensors" that cannot be modified
# in-place outside InferenceMode (e.g., later .zero_() in reset_state),
# which crashed train_and_save() at run_training():3627. no_grad behaviour
# is restored at call sites that need it.
def run_sc_diagnostics(
        net,
        *,
        centre_deg: float = 90.0,
        modality: Literal["A", "V", "B"] = "B",  # A = audio, V = visual, B = bimodal
        sigma_in: float = 5.0,
        stimulus_intensity: float = 1.0,
        pulse_frames: int = 5,
        n_frames: int = 20,
        noise_std: float = 0.0,
        verbose: bool = True,
) -> dict:
    """
    Passes a brief Gaussian pulse through *net* and returns a dictionary
    of quantitative measures plus an optional human‑readable print‑out.

    Parameters
    ----------
    centre_deg          – azimuth of the pulse (0–179°)
    modality            – 'A', 'V', or 'B' (= both modalities active)
    sigma_in            – input Gaussian σ (neurons) used for the stimulus
    stimulus_intensity  – scale factor applied to the Gaussian input
    pulse_frames        – how many external frames the pulse lasts
    n_frames            – total number of frames simulated
    noise_std           – additive Gaussian noise on the stimulus
    verbose             – if True, prints a nicely formatted report

    Returns
    -------
    metrics : dict
        {
          "spike_rates"      : {layer: Hz, ...},
          "currents"         : {"exc": …, "inh": …, "ampa": …, "nmda": …},
          "I_E_ratio"        : inh / exc,
          "STP"              : {"R_a_mean": …, "R_v_mean": …},
          "raw_time_series"  : defaultdict(list)       # (optional) per‑frame traces
        }
    """
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    n_sub = net.n_substeps
    dt_ms = net.dt
    N = net.n
    device = net.device

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    net.reset_state(batch_size=1)
    net._dbg_spk_A = net._dbg_spk_V = net._dbg_spk_MSI = 0.0
    net._dbg_steps = 0

    xA = torch.zeros(n_frames, N, device=device)
    xV = torch.zeros_like(xA)

    def _gauss_vec(center_deg):
        idx = int(round(center_deg * (N - 1) / (net.space_size - 1)))
        idx = max(0, min(N - 1, idx))
        xs = torch.arange(N, dtype=torch.float32, device=device)
        g = torch.exp(-0.5 * ((xs - idx) / sigma_in) ** 2)
        g = g * stimulus_intensity
        if noise_std > 0:
            g += torch.randn_like(g) * noise_std
        return g

    g_vec = _gauss_vec(centre_deg)

    if modality in ("A", "B"):
        xA[:pulse_frames] = g_vec
    if modality in ("V", "B"):
        xV[:pulse_frames] = g_vec

    valid = torch.ones(n_frames, 1, device=device, dtype=torch.bool)

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    ts_store = defaultdict(list)  # raw per‑frame traces (optional)

    for t in range(n_frames):
        _, _, _, _, sum_sM = net.update_all_layers_batch(
            xA[t].unsqueeze(0),
            xV[t].unsqueeze(0),
            valid_mask=None,
            return_spike_sum=True
        )

        # ---- mean currents (MSI excit) ----------------------------------
        I_M = net.I_M.detach()
        exc_curr = torch.clamp(I_M, min=0).mean().item()
        inh_curr = -torch.clamp(I_M, max=0).mean().item()

        mg_A = 1.0 / (1.0 + torch.exp(-net.mg_k * (net.v_dend_A - net.mg_vhalf)))
        mg_V = 1.0 / (1.0 + torch.exp(-net.mg_k * (net.v_dend_V - net.mg_vhalf)))
        I_nmda = (net.gNMDA
                  * net.nmda_m
                  * (mg_A + mg_V)
                  * (net.Erev_nmda - net.v_msi)).mean().item()
        ampa_curr = max(exc_curr - I_nmda, 0.0)  # safeguard floor

        ts_store["exc"].append(exc_curr)
        ts_store["inh"].append(inh_curr)
        ts_store["ampa"].append(ampa_curr)
        ts_store["nmda"].append(I_nmda)
        ts_store["MSI_spikes"].append(sum_sM.sum().item())

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    sim_time_s = n_frames * n_sub * dt_ms / 1_000.0  # seconds
    spike_rates = {
        "A": net._dbg_spk_A / (N * sim_time_s),
        "V": net._dbg_spk_V / (N * sim_time_s),
        "MSI": net._dbg_spk_MSI / (N * sim_time_s),
    }

    currents = {
        "exc": float(torch.tensor(ts_store["exc"]).mean()),
        "inh": float(torch.tensor(ts_store["inh"]).mean()),
        "ampa": float(torch.tensor(ts_store["ampa"]).mean()),
        "nmda": float(torch.tensor(ts_store["nmda"]).mean()),
    }
    currents["I_E_ratio"] = currents["inh"] / (currents["exc"] + 1e-12)

    stp_stats = {
        "R_a_mean": float(net.R_a.mean().item()),
        "R_v_mean": float(net.R_v.mean().item()),
    }

    metrics = dict(
        spike_rates=spike_rates,
        currents=currents,
        STP=stp_stats,
        raw_time_series=ts_store,
    )

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    if verbose:
        hdr = "=" * 72
        print(hdr)
        print("SC Multisensory‑Network Diagnostics")
        print(hdr)
        print(f"Stimulus  : {modality}  |  centre={centre_deg:.1f}°  |  σ={sigma_in} neur.")
        print(f"Intensity : {stimulus_intensity:.3f}  |  pulse={pulse_frames} frames")
        print(f"Sim time  : {sim_time_s * 1e3:.1f} ms  "
              f"({n_frames} frames × {n_sub} sub‑steps × {dt_ms:.1f} ms)")
        print("\n--- Mean firing rates (Hz) --------------------------------")
        for k, v in spike_rates.items():
            print(f"  {k:>4s}: {v:7.2f}")
        print("\n--- Membrane current @ MSI excit --------------------------")
        print(f"  Excitatory (all) : {currents['exc']:9.4f}")
        print(f"    – AMPA         : {currents['ampa']:9.4f}")
        print(f"    – NMDA         : {currents['nmda']:9.4f}")
        if currents['ampa'] > 0:
            print(f"      NMDA/AMPA    : {currents['nmda'] / currents['ampa']:.3f}")
        print(f"  Inhibitory (net) : {currents['inh']:9.4f}")
        print(f"  I/E ratio        : {currents['I_E_ratio']:.3f}")
        print("\n--- Short‑term plasticity resources -----------------------")
        print(f"  R_a (A→MSI) mean : {stp_stats['R_a_mean']:.3f}")
        print(f"  R_v (V→MSI) mean : {stp_stats['R_v_mean']:.3f}")
        print(hdr)

    return metrics


def analyze_late_nmda_vs_ampa(diagnostics, late_start=5):
    """
    diagnostics: the dict returned by run_sc_diagnostics(...),
                 which already contains raw_time_series in
                 diagnostics["raw_time_series"].

    late_start : int
        The time-step at which we start focusing on the NMDA fraction
        (e.g. skip the first 5 frames if you want).
    """
    ts = diagnostics["raw_time_series"]
    ampa_vals = ts["ampa"]  # or "exc" minus "nmda" if you prefer
    nmda_vals = ts["nmda"]
    steps = range(len(ampa_vals))
    plt.figure(figsize=(6, 4))
    plt.plot(steps, ampa_vals, label="AMPA current", color="C1")
    plt.plot(steps, nmda_vals, label="NMDA current", color="C0")

    # highlight or label the "late" region
    if late_start < len(ampa_vals):
        plt.axvspan(late_start, len(ampa_vals) - 1, color="gray", alpha=0.1,
                    label=f"Late window start={late_start}")

    plt.xlabel("External frame index")
    plt.ylabel("Mean Current (arbitrary units)")
    plt.title("AMPA vs. NMDA Over Time (SC Diagnostics)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    ampa_late = ampa_vals[late_start:]
    nmda_late = nmda_vals[late_start:]

    if len(ampa_late) == 0:
        print(f"No data after late_start={late_start}. Nothing to average.")
        return

    ampa_mean = sum(ampa_late) / len(ampa_late)
    nmda_mean = sum(nmda_late) / len(nmda_late)

    ratio = nmda_mean / (ampa_mean + 1e-9)

    print(f"[Late-window analysis] (t >= {late_start})")
    print(f"  AMPA mean: {ampa_mean:.3f},  NMDA mean: {nmda_mean:.3f}")
    print(f"  NMDA/AMPA ratio: {ratio:.3f}")


def _init(mat, mean):  # positive soft‑plus wrappers already active
    with torch.no_grad():
        mat.copy_(torch.abs(torch.randn_like(mat)) * mean)


def generate_two_event_offset_seq(loc, T=60, D=5, offset=0, space_size=180):
    """
    A positive offset → visual lags audio by <offset> macro steps (10 ms each);
    a negative offset → visual leads; 0 → simultaneous.
    """
    loc_seq = [999] * T
    mod_seq = ['X'] * T
    aud_on = 0 if offset >= 0 else abs(offset)
    vis_on = 0 if offset <= 0 else offset
    for t in range(aud_on, aud_on + D):
        loc_seq[t] = loc
        mod_seq[t] = 'A'
    for t in range(vis_on, vis_on + D):
        loc_seq[t] = loc
        mod_seq[t] = 'V' if mod_seq[t] == 'X' else 'B'
    return loc_seq, mod_seq


def generate_flash_sound_batch(
        offsets,
        loc=90,
        T=50,
        D=5,
        space_size=180
):
    loc_seqs = []
    mod_seqs = []
    offset_applied = []
    seq_lengths = []

    for off in offsets:
        seq_loc, seq_mod = generate_two_event_offset_seq(
            loc=loc, T=T, D=D, offset=off, space_size=space_size
        )
        loc_seqs.append(seq_loc)
        mod_seqs.append(seq_mod)
        offset_applied.append(False)
        seq_lengths.append(T)

    return loc_seqs, mod_seqs, offset_applied, seq_lengths


# Task #68: removed @torch.inference_mode() decorator. See note above
# run_sc_diagnostics for the same reasoning.
def run_temporal_integration(net, offsets, *, loc=90,
                             T=60, D=5, extra=5, stim_in=1,
                             log_charges=False):
    """
    Evaluate MSI population response for a range of AV onset offsets.

    Parameters
    ----------
    net      : trained MultiBatchAudVisMSINetworkTime
    offsets  : list/1-D array of int
               AV onset asynchronies in *macro-steps* (10 ms each).
               Positive  ->  visual lags audio.
               Negative  ->  visual leads audio.
    loc      : spatial location in degrees (default 90).
    T, D     : see generate_two_event_offset_seq  (T time-bins, D duration).
    extra    : number of *extra* macro-steps added to the integration
               window after the burst finishes (default 5 -> 50 ms).

    Returns
    -------
    dict with keys
      'spike_raster' : ndarray (T, len(offsets))       pop. spikes / 10 ms
      'int_spikes'   : 1-D ndarray (len(offsets),)     integrated counts
      'offsets_ms'   : list of onset offsets in ms
    """
    loc_seqs, mod_seqs, off_flags, seq_lens = generate_flash_sound_batch(
        offsets, loc=loc, T=T, D=D, space_size=net.space_size
    )
    max_len = max(seq_lens)
    xA, xV, mask = generate_av_batch_tensor(
        loc_seqs, mod_seqs, off_flags,
        n=net.n, space_size=net.space_size, sigma_in=net.sigma_in,
        noise_std=0.0, device=net.device, max_len=max_len, stimulus_intensity=stim_in,
    )

    # 2 .  run the network
    net.reset_state(len(offsets))
    rast = torch.zeros((max_len, len(offsets)), device=net.device)

    for t in range(max_len):
        net.update_all_layers_batch(xA[:, t], xV[:, t], mask[:, t])
        # population spike count (MSI excit.)
        rast[t] = net._latest_sMSI.sum(dim=1)

    # 3 .  integrate *aligned* windows
    int_spikes = []
    for i_off, off in enumerate(offsets):
        later_onset = abs(off)  # macro-steps until later stimulus
        win_start = later_onset
        win_stop = min(win_start + D + extra, rast.size(0))
        int_spikes.append(rast[win_start:win_stop, i_off].sum().item())

    return {
        'spike_raster': rast.cpu().numpy(),
        'int_spikes': np.asarray(int_spikes),
        'offsets_ms': [o * 10 for o in offsets]  # 1 macro-step = 10 ms
    }


from scipy.optimize import curve_fit


def fit_tbw_curve(offs_ms, int_spikes, *, model="gaussian", p0=None):
    """
    Fit a bell-shaped curve to the temporal-binding-window (TBW) points.

    Parameters
    ----------
    offs_ms     : 1-D array-like
        Audio–visual onset asynchronies in milliseconds.
    int_spikes  : 1-D array-like
        Integrated spike counts (same ordering as offs_ms).
    model       : "gaussian" | "flattop"
        Which analytical shape to fit.
    p0          : list or tuple, optional
        Initial parameter guesses.  If None, sensible defaults are chosen.

    Returns
    -------
    fit_dict    : dict
        {
          "xs"       : densely sampled x-axis,
          "ys"       : fitted curve evaluated at xs,
          "params"   : best-fit parameters,
          "cov"      : covariance matrix from curve_fit,
          "fwhm"     : full-width at half maximum (for Gaussian),
        }
    """
    offs = np.asarray(offs_ms, dtype=float)
    ints = np.asarray(int_spikes, dtype=float)

    # ----------- choose the analytic form -----------------------------------
    if model == "gaussian":
        def _f(x, base, amp, mu, sigma):
            return base + amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        if p0 is None:
            p0 = [ints.min(), np.ptp(ints), 0.0, 60.0]

    elif model == "flattop":
        def _f(x, base, amp, lc, lk, rc, rk):
            left = 1.0 / (1.0 + np.exp(-(x - lc) / lk))
            right = 1.0 / (1.0 + np.exp((x - rc) / rk))
            return base + amp * left * right

        if p0 is None:
            p0 = [ints.min(), np.ptp(ints), -80.0, 10.0, 80.0, 10.0]

    else:
        raise ValueError("model must be 'gaussian' or 'flattop'")

    # ----------- non-linear least-squares fit --------------------------------
    popt, pcov = curve_fit(_f, offs, ints,
                           p0=p0)  # SciPy’s LM/Trust-Region optimiser :contentReference[oaicite:0]{index=0}

    xs = np.linspace(offs.min(), offs.max(), 600)
    ys = _f(xs, *popt)

    fwhm = None
    if model == "gaussian":
        sigma = popt[3]
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma  # standard formula :contentReference[oaicite:1]{index=1}

    return {"xs": xs, "ys": ys, "params": popt, "cov": pcov, "fwhm": fwhm}


def plot_temporal_binding(results, *, fit_model="gaussian", **fit_kw):
    """
    Visualise the spike raster AND the TBW curve with an analytical fit,
    *and* print the key numerical values so they can be logged or pasted.

    Parameters
    ----------
    results : dict
        Output of run_temporal_integration.
    fit_model : {"gaussian", "flattop", None}
        Which model to super‑impose.  Pass None to disable fitting.
    fit_kw : dict
        Extra keywords forwarded to fit_tbw_curve.
    """
    rast = results["spike_raster"]
    ints = results["int_spikes"]
    offs = np.asarray(results["offsets_ms"])

    # —–––––––––––––––– heat‑map panel ––––––––––––––––––
    plt.figure(figsize=(8, 4))
    plt.imshow(rast,
               origin="lower", aspect="auto",
               extent=[offs[0], offs[-1], 0, 10 * rast.shape[0]])
    plt.colorbar(label="MSI pop‑spikes / 10 ms")
    plt.xlabel("Audio – Visual onset (ms)")
    plt.ylabel("Time (ms)")
    plt.title("MSI activity vs. AV asynchrony")

    # —–––––––––––––––– binding curve –––––––––––––––––––
    plt.figure(figsize=(4, 3))
    plt.plot(offs, ints, "o", label="data")

    if fit_model is not None:
        fit = fit_tbw_curve(offs, ints, model=fit_model, **fit_kw)
        plt.plot(fit["xs"], fit["ys"], "-", lw=2, label=f"{fit_model} fit")
        # annotate peak & width for Gaussian
        if fit_model == "gaussian":
            base, amp, mu, sigma = fit["params"]
            fwhm = fit["fwhm"]
            plt.annotate(
                f"μ = {mu:+.0f} ms\nFWHM = {fwhm:.0f} ms",
                xy=(mu, base + amp),
                xytext=(mu + 30, base + 0.6 * amp),
                arrowprops=dict(arrowstyle="->", lw=0.8),
                fontsize=8,
            )

    plt.axvline(0, ls="--", c="k", lw=0.7)
    plt.xlabel("Audio – Visual onset (ms)")
    plt.ylabel("Integrated spikes (0–100 ms)")
    plt.title("Temporal binding window")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.show()

    print("Offsets (ms) :", offs.tolist())  # diagnostics
    print("Int. spikes  :", ints.tolist())  # diagnostics


def run_training(
        batch_size=1000,
        n_unsup_epochs=80,
):
    """
    Main training run
    """
    print(f"Initializing network with batch size {batch_size}...")
    net = MultiBatchAudVisMSINetworkTime(
        n_neurons=180,
        batch_size=batch_size,
        lr_unimodal=2e-2,
        lr_msi=2e-2,
        lr_readout=8e-4,
        sigma_in=10.0,
        sigma_teacher=2.0,  # (not used directly in final, replaced by scheduling)
        noise_std=0.02,
        single_modality_prob=0.5,
        v_thresh=0.3,
        dt=0.1,
        tau_m=20.0,
        n_substeps=100,
        loc_jitter_std=0,
        space_size=180,
        # task #128: REVERTED task #94 INT-1 (100 / 260). Pristine fb6d3f6
        # run_training set these to 250 / 400 substeps == 25 / 40 ms at dt=0.1
        # (biologically plausible A/V SC conduction delays).
        conduction_delay_a2msi=250,
        conduction_delay_v2msi=400
    )

    init_W_inA = net.W_inA.clone().cpu().numpy()
    init_W_inV = net.W_inV.clone().cpu().numpy()

    net.set_inhib_plasticity(True)

    assign_unimodal_preferred_locations(net)

    print("  W_inA_inh sum =", net.W_inA_inh.sum().item())
    print("  W_inV_inh sum =", net.W_inV_inh.sum().item())
    print("  W_a2msiInh_AMPA sum =", net.W_a2msiInh_AMPA.sum().item())
    print("  W_a2msiInh_NMDA sum =", net.W_a2msiInh_NMDA.sum().item())
    print("  W_v2msiInh_AMPA sum =", net.W_v2msiInh_AMPA.sum().item())
    print("  W_v2msiInh_NMDA sum =", net.W_v2msiInh_NMDA.sum().item())
    print("  W_msiInh2Exc_GABA sum =", net.W_msiInh2Exc_GABA.sum().item())
    print("  W_MSI_inh sum =", net.W_MSI_inh.sum().item())
    print("  g_GABA =", net.g_GABA)

    net.b_uniA.data.fill_(0.0)
    net.b_uniV.data.fill_(0.0)

    _init(net.W_a2msi_AMPA, 0.004)
    _init(net.W_v2msi_AMPA, 0.004)
    _init(net.W_a2msi_NMDA, 0.004)
    _init(net.W_v2msi_NMDA, 0.004)

    with torch.no_grad():

        # Task #16/#27: gNMDA recalibrated from legacy 0.05 to 1.30 to compensate
        # for the (1 - exp(-dt/tau_syn)) ≈ dt/tau_syn factor introduced by the
        # dt-correct NMDA injection (Form 2). Empirically calibrated at dt=0.1
        # on M00 fixed-seed; preserves paper TBW HW = 107 ms control.
        net.gNMDA = 1.30
        net.tau_nmda = 80.0
        net.nmda_alpha = 0.1
        net.Erev_nmda = 20.0
        net.tau_nmdaVolt = 100.0
        net.v_nmda_rest = -65.0
        net.nmda_vrest_offset = 7.0
        net.mg_vhalf = -35.0


    net.u_a.fill_(0.7)
    net.u_v.fill_(0.7)
    net.tau_rec = 400.0

    print("[tune_for_biology] coarse biological calibration applied")

    # task #106 (researcher #105): remove the broken calibrator (#101/#103) and
    # use a manual input_scaling=400 — paired with the tighter target_mean=0.00006
    # on the MSI-input AMPA/NMDA weights (Training.py:2483-2488). g_FFinh starts
    # at 0.6 and AGC takes over from there during STDP.
    net.input_scaling = 400
    net.g_FFinh = 0.6
    net.g_GABA = 10



    # Unsupervised STDP - Useless, no need
    print("\n--- STDP training (unsupervised) ---")
    unsup_start = time.time()
    last_ep = 0
    for epoch in range(n_unsup_epochs):
        last_ep = epoch
        epoch_start = time.time()
        if 2 <= epoch <= 79:  # choose any window you like
            if net._probe is None:
                net._probe = AMPANMDADebugger()
            else:
                net._probe.reset()
        with torch.no_grad():
            W_before = net.W_inA.clone()  # snapshot *before* training

        net.train_unsupervised_batch(1000, batch_size=256, debug=False, epoch_idx=epoch)  # run some sequences
        net.print_epoch_spike_summary(f"unsup {epoch + 1:02d}")

        if 2 <= epoch <= 79:
            net._probe.report(net, f"epoch {epoch}")

        with torch.no_grad():
            delta = (net.W_inA - W_before).abs().max().item()
            print("Δ‖W_inA‖ =", delta)
        epoch_time = time.time() - epoch_start
        print(f"  Unsup Epoch {epoch + 1}/{n_unsup_epochs} - Time: {epoch_time:.2f}s")
    unsup_time = time.time() - unsup_start
    print(f"Unsupervised training completed in {unsup_time:.2f}s")

    net.set_inhib_plasticity(False)

    ckpt = make_checkpoint(net,
                           epoch=last_ep,
                           optim=None,  # or None if you’re done training
                           comment="MSI model – paper Figure 3")

    save_path = Path("checkpoint") / "msi_redone_agc_fix_.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"✅  Full checkpoint written to  {save_path.resolve()}")

    net = None

    ckpt_path = Path("checkpoint/msi_redone_agc_fix_.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"])
    for k, v in ckpt["mutable_hparams"].items():
        setattr(net, k, v)
    if "torch_cpu" in ckpt:
        torch.set_rng_state(ckpt["torch_cpu"])
    if "torch_cuda" in ckpt and ckpt["torch_cuda"] is not None:
        torch.cuda.set_rng_state(ckpt["torch_cuda"])

    net.eval()  # or net.train() to keep learning
    print("✔️  Model ready for evaluation.")

    net.reset_state()

    net.set_inhib_plasticity(True)

    net.reset_state()


    offsets = list(range(-50, 51))  # −100 … +100 ms in 10 ms steps
    res_ti = run_temporal_integration(net, offsets, loc=90, T=60, D=5, stim_in=1)
    plot_temporal_binding(res_ti)
    net.reset_state()


    msi_activity_summary(net,
                         centre_deg=90,
                         sigma_in=5,
                         pulse_len=15,
                         n_steps=30,
                         modality="A",
                         style="ggplot")

    msi_activity_summary(net,
                         centre_deg=130,
                         sigma_in=5,
                         pulse_len=15,
                         n_steps=30,
                         modality="V",
                         style="ggplot")

    return {
        'network': net

    }


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def train_and_save(model_idx: int,
                   base_seed: int = 42,
                   out_dir: str = "checkpoint") -> Path:
    """
    Build ➜ train ➜ checkpoint one network replica.

    Parameters
    ----------
    model_idx : 0‑based integer label (0…9)
    base_seed : deterministic offset so each net sees a unique RNG stream
    out_dir   : folder where .pt files are written

    Returns
    -------
    Path to the file that was saved.
    """
    torch.manual_seed(base_seed + model_idx)
    np.random.seed(base_seed + model_idx)
    results = run_training()
    net = results["network"]
    ckpt = make_checkpoint(net,
                           epoch=0,
                           comment=f"replica {model_idx}",
                           rng_tag=True)

    # save to disk
    out_path = Path(out_dir) / f"msi_model_surr_16_{model_idx:02d}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    del net
    torch.cuda.empty_cache()

    return out_path


# ----------------------------------------------------------------------
if __name__ == "__main__":
    N_REPLICAS = 10
    saved = []
    for i in range(N_REPLICAS):
        print(f"\n=== TRAINING REPLICA {i + 1}/{N_REPLICAS} ===")
        path = train_and_save(i)
        saved.append(path)
        print(f"✔ Saved checkpoint ➜ {path}")
    print("\nAll replicas finished:")
    for p in saved:
        print("  •", p)

