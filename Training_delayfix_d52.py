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
  - Disynaptic inhibition only: unimodal -> MSI_inh interneurons ->
    MSI_exc GABA (task#192 Phase A removed the direct A/V -> MSI_exc
    feed-forward inhibitory shortcut as biologically unjustified).
  - Lateral / surround inhibition in MSI_exc (Mexican-hat geometry).
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
  - ``gNMDA``                   — global NMDA conductance gain.
  - ``g_GABA``                  — MSI_inh -> MSI_exc GABA conductance.
  - ``input_scaling`` (150.0)   — input current scaling.
  - Note: ``g_FFinh`` is retained as an orphan attribute (init still sets
    a value; SBW/TBW probes save/restore it) but the direct FF-inh
    injection it gated was removed in task#192 Phase A. It no longer
    affects network dynamics.
  - AGC loops were removed in task#185 (minimal-arch). iSTDP / Hebbian
    rules on plastic edges now provide E/I regulation; ``plasticity_enabled``
    is still used to suppress weight updates in eval paths.

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
import os
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
                                 temporal_jitter_max=0,
                                 sigma_dL_frames=0.0):
    """
    Create synthetic A/V sequences with spatial offsets and optional temporal jitter.

    Parameters
    ----------
    temporal_jitter_max : int
        Max A/V onset offset in frames (0 disables jitter). This is the legacy
        per-EVENT INDEPENDENT onset jitter (uncorrelated draw for every bimodal
        event) — left untouched for byte-identity.
    sigma_dL_frames : float
        task #100 — std (in 10 ms macro-frames) of the per-trial COMMON-MODE
        A-V latency jitter ΔL. ONE shared Gaussian draw per trial shifts the
        whole trial's A-V relative onset together (correlated by construction,
        unlike temporal_jitter_max). 0.0 disables it AND draws nothing -> the
        numpy RNG stream is untouched -> byte-identical to the pristine build.
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

        # task #100 — ONE shared common-mode A-V latency jitter ΔL per trial (gated; integer frames).
        # Knob OFF (sigma_dL_frames == 0) draws NOTHING -> numpy RNG stream untouched -> byte-identical.
        dL_trial = 0
        if sigma_dL_frames and sigma_dL_frames > 0.0:
            dL_trial = int(round(np.random.normal(0.0, float(sigma_dL_frames))))

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

                # task #100 — fold the per-trial COMMON-MODE ΔL into the A-V relative onset
                # (shared dL_trial across the whole trial; integer-frame; matches the
                # measurement-path edit + the #98-H4 kernel). Clamped to the in-window room so
                # BOTH bursts are always placed (no lost modality, no truncation bias).
                # dL_trial == 0 (knob off) -> rel == dt_frames -> byte-identical to pristine.
                rel = dt_frames
                if mode_tag == 'B' and dL_trial != 0:
                    rel = dt_frames + dL_trial
                    _msh = max(0, (T - D) - t)
                    if abs(rel) > _msh:
                        rel = int(np.sign(rel) * _msh)

                a_start = t + (-rel if rel < 0 else 0)
                v_start = t + (rel if rel > 0 else 0)
                window_len = D + abs(rel)

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

    G = net._get_cached_gaussian_kernel_sharpened(sigma)  # (n,n) — cached; #120 afferent-sharpen (f=1.0 => exact original)

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


def apply_topographic_anchor_msi_inh(net, layer="A", lr=1e-4, sigma=3.0):
    """Hebbian/Oja topographic anchoring for A/V -> MSI_inh feedforward weights.

    task #192 Phase D (sci_inhibition_functional_role.md §5.4a):
    Mirror of apply_topographic_anchor_msi but for the INPUT->INTERNEURON
    edges (W_a2msiInh_AMPA/NMDA, V mirror). The interneuron is at the same
    depth in the network as MSI excit; the input synapse onto it is
    excitatory (AMPA+NMDA), so the standard exc Hebbian+Oja rule applies
    (D'Amour-Froemke 2015 puts iSTDP on the OUTPUT GABA synapse, not the
    excitatory input drive; see §4.3).

    Weight shape (n_inh, n): post-interneuron i, pre-input j.
    Kernel shape (n_inh, n): cross-population Gaussian over evenly-spaced
    positions on the n -> n_inh map projection.
    """
    if layer == "A":
        W_AMPA = net.W_a2msiInh_AMPA
        W_NMDA = net.W_a2msiInh_NMDA
        attr_AMPA = "W_a2msiInh_AMPA"
        attr_NMDA = "W_a2msiInh_NMDA"
    else:
        W_AMPA = net.W_v2msiInh_AMPA
        W_NMDA = net.W_v2msiInh_NMDA
        attr_AMPA = "W_v2msiInh_AMPA"
        attr_NMDA = "W_v2msiInh_NMDA"

    s_post = net._latest_sMSI_inh           # (B, n_inh)
    r_post = s_post.mean(dim=0)             # (n_inh,)

    G = net._get_cached_cross_gaussian_kernel(sigma, net.n_inh, net.n)  # (n_inh, n)
    dW_ampa = lr * (r_post.unsqueeze(1) * G)               # (n_inh, n)
    dW_ampa_decay = lr * (r_post.unsqueeze(1) * W_AMPA)    # (n_inh, n)
    net._p_add(attr_AMPA, dW_ampa - dW_ampa_decay)
    dW_nmda = lr * (r_post.unsqueeze(1) * G)
    dW_nmda_decay = lr * (r_post.unsqueeze(1) * W_NMDA)
    net._p_add(attr_NMDA, dW_nmda - dW_nmda_decay)


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
                          dt_minutes: float = 1.67e-5):  # task#180 Phase1: biology timescale (was 1.0; bug: 60000× too fast)
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
        # task #99 — correlated trial-timing noise (common-mode A-V latency jitter) for the TBW
        # graded-bell fix. Gated net-level knob, DEFAULT 0.0 -> byte-identical to pristine (23326edb).
        # Units: macro-step frames (10 ms each); 3.0 frames = 30 ms (researcher dossier σ_ΔL).
        # Consumed on the measurement path via generate_two_event_offset_seq(..., sigma_dL_frames=...);
        # the training-path wiring (generate_event_loc_seq_batch) is the retrain-phase edit.
        self.sigma_dL_frames = float(os.environ.get('SIGMA_DL_FRAMES', '0.0'))

        # task #327 — INDEPENDENT per-channel afferent-latency (conduction-delay) jitter, σ in MS.
        # DISTINCT from sigma_dL_frames (macro-frame, common-mode, correlated A-V offset): this is
        # per-CHANNEL (A,V independent), per-TRIAL, SUBSTEP-resolution (dt=0.1ms) arrival jitter on the
        # conduction-delay ring-buffer READ, so it graduates the deterministic first-spike-latency STEP
        # into biology's graded inverse-effectiveness taper. GATED, DEFAULT 0.0 -> no draw (RNG untouched)
        # -> buffers unchanged (length==delay) -> read collapses to buf[pos] -> BYTE-IDENTICAL to pristine.
        self.afferent_jitter_ms = float(os.environ.get('AFFERENT_JITTER_MS', '0.0'))
        # per-trial jitter state (populated by reset_state when ON; None => scalar buf[pos] read path).
        self._aff_jitter_a = None
        self._aff_jitter_v = None

        # task #120 — afferent receptive-field SHARPENING baked into TRAINING (#118 dossier).
        # Gated net-level knob, DEFAULT 1.0 -> byte-identical to the pristine build. f<1.0 narrows
        # the UNIMODAL topographic-anchor Gaussian sigma -> sigma*f AND scales the target kernel by
        # 1/f so each column-sum (total afferent charge per unit) is conserved (narrower-but-taller;
        # Shah&Crair area×peak invariant, #118 §3). Consumed ONLY by apply_topographic_anchor_unimodal
        # via _get_cached_gaussian_kernel_sharpened, so the afferent weights converge to a sharper,
        # drive-conserved RF during training (present in train AND, via the trained weights, in measure).
        # Scope: unimodal W_inA/W_inV only (A/V->MSI and ->MSI_inh anchors untouched).
        self.afferent_sharpen_factor = float(os.environ.get('AFFERENT_SHARPEN_FACTOR', '1.0'))

        # task #132: gated d0≈15° intermediate lateral-inhibitory SURROUND knob, DEFAULT budget 0.0 ->
        # byte-identical (no ring added). budget>0 ADDS a fixed structural Gaussian inhibitory ring at
        # ±d0_deg on TOP of the intact near-field W_MSI_inh (ring=0 at the bump center so the single
        # focal bump is untouched; ring total weight = budget × W_MSI_inh.sum()), reproducing the #131
        # debugger-proven SBW lever EXACTLY. W_MSI_inh is non-plastic (requires_grad=False, never _p_add'd,
        # iSTDP stripped) so the ring is a PERMANENT structural feature the plastic pathways co-adapt
        # AROUND during the retrain (bake-into-train, present in train AND measure). Confirmed operating
        # point: budget=1.0, d0=15°, w_idx=4 idx (= R_near). Applied once at init below, after W_MSI_inh
        # exists, using the post-Positive() sum so it matches the proven inference geometry.
        self.surround_budget = float(os.environ.get('SURROUND_BUDGET', '0.0'))
        self.surround_d0_deg = float(os.environ.get('SURROUND_D0_DEG', '15.0'))
        self.surround_w_idx = float(os.environ.get('SURROUND_W_IDX', '4.0'))

        self.pv_nmda = 5.0  # task#185: kept per Q-AGC-5 audit (variable retained even though AGC use removed)
        # task #185 (minimal-arch §5.1): REMOVED self.targ_ratio = 5.0
        # (AGC fast-loop setpoint; AGC removed entirely).

        self.W_latA = torch.zeros((self.n, self.n), device=self.device)
        self.W_latV = torch.zeros((self.n, self.n), device=self.device)
        self.g_latA = 0.1  # Lateral inhibition gain for A
        self.g_latV = 0.1  # Lateral inhibition gain for V
        self._probe = AMPANMDADebugger()  # ← add near other debug fields
        self.enable_probe = False  # opt-in: set True to collect AMPA/NMDA stats
        self._ei_record = None  # E/I component recording (None = off)

        self.tau_ampa_lp = 2.5  # ms  (same as self.tau_syn)
        self.ampa_alpha = 1.0  # scale factor per injection
        self.gAMPA_LP = float(os.environ.get("GAMPA_LP", 1.0))  # gain when converting ampa_m → current
        # task#130 Form-A: dVm/dt-ADAPTIVE spike threshold (Azouz & Gray 2000; #106/#125 latency fix). Env knobs,
        # all defaulting to a BYTE-IDENTICAL no-op (k_dvdt=0 => _formA_on False => original static +30mV threshold).
        #   v_thresh_eff = clamp(spike_threshold - k_dvdt*lowpass(relu(dV/dt), tau_dvdt), v_thresh_floor, spike_threshold)
        self.k_dvdt = float(os.environ.get("K_DVDT", 0.0))            # threshold-lowering slope (ms); reasoned start 0.3, biology-anchored to a 2-4mV drop
        self.tau_dvdt = float(os.environ.get("TAU_DVDT", 3.0))        # low-pass window (ms) for the relu(dv/dt) trace (A&G 1-5ms)
        self.v_thresh_floor = float(os.environ.get("V_THRESH_FLOOR", 20.0))  # clamp floor (mV); max 10mV drop below the +30mV peak
        self.dvdt_cap = float(os.environ.get("DVDT_CAP", 50.0))       # cap on relu(dv/dt) (mV/ms) so the regenerative upstroke can't inflate the trace
        self._formA_on = (self.k_dvdt != 0.0)                         # toggle: OFF => static threshold (byte-identical); re-derived after mh-restore in val36 load_ckpt
        self.dvdt_trace_msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)  # low-pass relu(dv/dt) state (per-MSI); reset per trial
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

        # task #192 Phase A (sci_inhibition_functional_role.md §5.4d, §1):
        # REMOVED W_inA_inh / W_inV_inh entirely. The "direct unimodal -> MSI_exc
        # inhibitory shortcut" they parameterised is biologically unjustified
        # (no anatomical correlate in cat/primate SCi: input pathways are
        # excitatory; inhibition is supplied by GABAergic interneurons via the
        # disynaptic A/V -> MSI_inh -> MSI_exc loop). Disynaptic inhibition is
        # retained via W_a2msiInh_AMPA/NMDA -> W_msiInh2Exc_GABA below.

        init_a2msi_inh = torch.tensor(0.005 * np.random.randn(self.n_inh, self.n),
                                      dtype=torch.float32, device=self.device)
        init_v2msi_inh = torch.tensor(0.005 * np.random.randn(self.n_inh, self.n),
                                      dtype=torch.float32, device=self.device)

        # FF-inhibition recruitment lever (debugger DIAG_C1V48_POP_MISMATCH): scale the A/V->MSI_inh
        # interneuron INPUT init so weak coincidence leaves the interneuron sub-threshold (weak-drive IE
        # escape / divisive-normalisation-with-threshold). env-config; default 1.0 == original. Multiplies
        # the init AFTER the torch.rand draw, so the RNG sequence is unchanged (byte-identical at 1.0).
        self.msiInh_input_init_scale = float(os.environ.get("MSIINH_INPUT_SCALE", 1.0))
        # task#102 (B): PERSISTENT forward-multiplier on the FFI-recruitment weights (W_*2msiInh), applied EVERY forward
        # step (not just init) => washout-PROOF, faithful to debugger#92's frozen forward-multiply (offset-E/I 0.481).
        # Default 1.0 == byte-identical. Sibling to the init-only msiInh_input_init_scale above; saved to
        # mutable_hparams + restored by load_ckpt (g_GABA/mg_k-style) so MEASURE honors the trained recruitment.
        self.msiInh_input_gain = float(os.environ.get("MSIINH_INPUT_GAIN", 1.0))

        self.W_a2msiInh_AMPA = nn.Parameter(self.msiInh_input_init_scale * 0.005 * 5.0 * torch.rand(self.n_inh, self.n, device=self.device),
                                            requires_grad=False)
        self.W_a2msiInh_NMDA = nn.Parameter(self.msiInh_input_init_scale * 0.005 * 15.0 * torch.rand(self.n_inh, self.n, device=self.device),
                                            requires_grad=False)
        self.W_v2msiInh_AMPA = nn.Parameter(self.msiInh_input_init_scale * 0.005 * 5.0 * torch.rand(self.n_inh, self.n, device=self.device),
                                            requires_grad=False)
        self.W_v2msiInh_NMDA = nn.Parameter(self.msiInh_input_init_scale * 0.005 * 15.0 * torch.rand(self.n_inh, self.n, device=self.device),
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
        self.rho0 = 4.5  # task#150/#156: trace units; target_F=30Hz (= rho0 × 6.667)
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
        # task #185 (minimal-arch §5.1): REMOVED AGC cadence + gate state
        # (T_AGC_FAST_MS, T_AGC_SLOW_MS, _last_agc_fast_t, _last_agc_slow_t).
        # Engineering hack removed; iSTDP is the biology-conformant E/I regulator.

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

        # ------------- iSTDP traces (Phase E) --------------
        # task #192 Phase E (sci_inhibition_functional_role.md §5.4b):
        # D'Amour-Froemke 2015 symmetric +/-10 ms Hebbian iSTDP on the
        # interneuron -> MSI_exc GABA edge (W_msiInh2Exc_GABA).
        # tau ~ 20 ms covers the +/-10 ms biological half-window
        # (Endo 1998 / D'Amour-Froemke 2015 mouse aud-cortex L5).
        # Per spec equation: dW = eta * ((post_trace - base) * pre_spike
        #                                + (pre_trace - base) * post_spike)
        # with the trace baseline subtraction giving net LTD for long ISIs.
        self.pre_trace_msiInh2Exc = torch.zeros(
            (self.batch_size, self.n_inh), dtype=torch.float32, device=self.device
        )
        self.post_trace_msiInh2Exc = torch.zeros(
            (self.batch_size, self.n), dtype=torch.float32, device=self.device
        )
        # Increment 3: recurrent MSI->MSI excitatory STDP eligibility traces.
        # Same (batch, n) shape + machinery as the FF->MSI excitatory STDP traces
        # (pre_trace_a2msi etc.); here pre == post == the MSI population. Re-zeroed
        # in reset_state, mirroring the FF traces exactly.
        self.pre_trace_msi_rec = torch.zeros(
            (self.batch_size, self.n), dtype=torch.float32, device=self.device
        )
        self.post_trace_msi_rec = torch.zeros(
            (self.batch_size, self.n), dtype=torch.float32, device=self.device
        )
        # delay-fix: previous external-step MSI spikes = the DELAYED pre for the
        # recurrent STDP (pre leads post by 1 external step). None => zeros at the
        # start of each sequence; re-Noned in reset_state.
        self._prev_sMSI_rec = None
        self.tau_istdp_pre = 20.0    # ms — symmetric pre-trace decay
        self.tau_istdp_post = 20.0   # ms — symmetric post-trace decay
        self.eta_istdp = 2e-5        # was 5e-4 (25× lower); debugger #193 §Q3.E range 1e-5 to 5e-5
        self.istdp_baseline = float(os.environ.get("ISTDP_BASELINE", 0.6))    # was 0.0 (bug); trace_ss at ρ=30 Hz × τ_pre = 30·0.02 = 0.6
                                     # task#310 inhibition-setpoint lever: env-exposed ISTDP_BASELINE (default 0.6 == byte-identical). Lower baseline ->
                                     # lower iSTDP target rate -> settles HIGHER W_msiInh2Exc_GABA (active inline iSTDP subtracts it every step).
                                     # biology: Stein-Stanford 2008 cat SC sustained 10-30 Hz
        self.W_gaba_clamp = 0.5      # upper bound on W_msiInh2Exc_GABA per spec

        # ------------- Izhikevich params --------------
        # For unimodal excit
        self.aA, self.bA, self.cA, self.dA = 0.02, 0.2, -65.0, 8.0
        self.aV, self.bV, self.cV, self.dV = 0.02, 0.2, -65.0, 8.0
        # MSI excit
        self.aM, self.bM, self.cM, self.dM = 0.02, 0.2, -65.0, 8.0
        # task#104 (decouple retrain): spike-triggered adaptation knobs for MSI_exc, env-exposed (defaults byte-identical).
        # bM = subthreshold recovery coupling (u_msi=bM*v_msi); dM = spike-triggered increment of the recovery/adaptation
        # variable (u_msi += dM on spike). Raising dM/bM strengthens adaptation -> suppresses sustained/2nd-event firing.
        self.bM = float(os.environ.get("ADAPT_BM", self.bM))
        self.dM = float(os.environ.get("ADAPT_DM", self.dM))
        # task#288 recovery-timescale lever: aM = adaptation recovery rate (tau_adapt=1/aM); env ADAPT_A, default 0.02 byte-identical.
        self.aM = float(os.environ.get("ADAPT_A", self.aM))
        # MSI inh (fast spiking)
        self.aMi, self.bMi, self.cMi, self.dMi = 0.1, 0.2, -65.0, 2.0
        # task#376 E/I-specific adaptation lever (debugger #375): interneuron (MSI_inh) SFA env-exposed,
        # defaults byte-identical (aMi=0.1 -> tau_adapt_inh=1/aMi ~ 10 ms; dMi=2.0). MORE-sustained inh
        # adaptation (aMi DOWN / dMi UP) keeps feedforward inhibition persistent at temporal offsets ->
        # narrows the exc fused window TEMPORALLY at fixed exc aM=0.008 (protects dRACE/G6). aMi/dMi are
        # already saved to mutable_hparams + restored by val36.load_ckpt so MEASURE matches training.
        self.aMi = float(os.environ.get("ADAPT_A_INH", self.aMi))
        self.dMi = float(os.environ.get("ADAPT_DM_INH", self.dMi))
        self.msi_inh_refrac_substeps = 20  # 2 ms @ dt=0.1ms — absolute refractory ceiling for PV/MSI_inh
        # -- soft-bound FF->MSI_exc STDP config (graded MSI_exc; no content-free brake) --
        # Weight-dependent STDP for the 4 FF->MSI_exc weights: dW=lr*[A+*(Wmax-W)^mu - A-*W^mu].
        # Validated in the debugger soft-bound stint (dbg6); per-synapse caps, mu=1.0.
        # Unimodal W_inA/W_inV are NOT in this dict -> they stay additive (intended).
        self._softbound_mu = 1.0
        self._softbound_wmax = {
            'W_a2msi_AMPA': 0.006, 'W_a2msi_NMDA': 0.018,
            'W_v2msi_AMPA': 0.006, 'W_v2msi_NMDA': 0.018,
        }
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
        self.msi_inh_refrac = torch.zeros((self.batch_size, self.n_inh), device=self.device)

        # Out
        self.v_out = torch.full((self.batch_size, self.n), self.cO, device=self.device)
        self.u_out = self.bO * self.v_out

        # ------------- Spikes from last substep --------------
        # task#114 STEP-1: substep-resolution first-spike probe state (measurement-only; default OFF so every
        # other measure path AND training stay byte-identical). _fs_probe_on persists across reset_state; the
        # per-trial first-spike-substep buffer + global substep clock are (re)initialised here + in reset_state.
        self._fs_probe_on = False
        self._first_spike_substep = torch.full((self.batch_size,), -1, dtype=torch.long, device=self.device)
        self._fs_substep_clock = 0
        self._fs_dvdt_trace = torch.full((self.batch_size,), float('nan'), dtype=torch.float32, device=self.device)  # task#130 anchor: dV/dt trace at the crossing substep (probe+Form-A only)
        self._v_msi_substep_trace = []   # latency #152: substep-res max-v_msi trace (PRE-reset peak; appended iff _fs_probe_on, cleared per trial; logging-only => byte-identical when off). Read via torch.stack(net._v_msi_substep_trace).
        self._latest_sA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self._latest_sOut = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)

        # ------------- Synaptic currents --------------
        # task #192 Phase B (sci_inhibition_functional_role.md §5.2): SPLIT
        # tau_syn into tau_ampa (2.5 ms, fast glutamate — unchanged) and
        # tau_gaba (50 ms — biology IPSC weighted tau_w, Sergeeva 2006 mouse
        # SC median, midway between fast cortical 22 ms and slow EGFP+ 132 ms).
        # tau_syn kept as alias = tau_ampa for legacy code paths (ckpt save,
        # decay_factor in update_all_layers_batch).
        self.tau_ampa = 2.5    # ms
        self.tau_gaba = float(os.environ.get('TAU_GABA', '50.0'))   # ms — task#P6 env override (mirrors trainer so measurement matches the trained value; read live into gaba_decay)
        self.pv_gaba_scale = float(os.environ.get('PV_GABA_SCALE', '1.0'))  # ASD lever: disynaptic PV(MSI_inh)->MSI_exc GABA conductance scale; 1.0 = NT byte-identical. MEASURE mirrors trainer; val36.load_ckpt restores the trained value from the ckpt so measurement matches training.
        # surround-shunt GABA (revised IE spec): convert ONLY the lateral surround to a divisive/shunting conductance; OFF default => byte-identical
        self.E_gaba = float(os.environ.get('E_GABA', '-70.0'))                  # mV, GABA-A reversal on the Izhikevich scale
        self.k_shunt_surr = float(os.environ.get('K_SHUNT_SURR', '0.0'))        # 1/mV conductance->current scale (charge-matched; debugger supplies)
        self.gaba_shunt_surr = (os.environ.get('GABA_SHUNT_SURR', '0') == '1')  # OFF default => subtractive surround (byte-identical)
        self.tau_syn = self.tau_ampa  # legacy alias (AMPA-side decay)
        self.I_A = torch.zeros((self.batch_size, self.n), device=self.device)
        self.I_V = torch.zeros((self.batch_size, self.n), device=self.device)
        self.I_M = torch.zeros((self.batch_size, self.n), device=self.device)
        # task #192 Phase B: NEW separate GABAergic current state for MSI exc.
        # Accumulates tonic-GABA + disynaptic-GABA (MSI_inh->MSI_exc) +
        # surround-GABA (Mexican-hat lateral) injections, decays at tau_gaba.
        # Subtracted from I_M at the Izhikevich integration step so the slow
        # IPSC time-course is represented faithfully without mixing into the
        # fast AMPA/NMDA-driven I_M.
        self.I_M_gaba = torch.zeros((self.batch_size, self.n), device=self.device)
        # surround-shunt: separate accumulator for the lateral-surround GABA when GABA_SHUNT_SURR=1; stays 0 when OFF => byte-identical
        self.I_M_gaba_surr_sh = torch.zeros((self.batch_size, self.n), device=self.device)
        self._last_I_surr_shunt = torch.zeros((self.batch_size, self.n), device=self.device)  # task #55: last-substep surround current, exposed for shunt-aware E/I; 0 when OFF
        self.I_M_inh = torch.zeros((self.batch_size, self.n_inh), device=self.device)
        self.I_O = torch.zeros((self.batch_size, self.n), device=self.device)

        self.I_ampa_filtered = torch.zeros((self.batch_size, self.n), device=self.device)

        # ------------- Conduction delay buffers --------------
        # task #192 Phase C (sci_inhibition_functional_role.md §5.3):
        # OVERRIDE the constructor kwargs `conduction_delay_a2msi` /
        # `_v2msi` with biology-anchored values. Whyland-Bickford 2018
        # mouse iSC: EPSC ~4 ms, disynaptic IPSC ~9.25 ms (+5.2 ms over EPSC).
        # At dt=0.1 ms: 4 ms = 40 substeps, 5.2 ms over = 52 substeps.
        # The previous legacy values (250 / 400 substeps = 25 / 40 ms)
        # over-stated cat/mouse SC conduction by 6-10x; the §5.3 anchor
        # tightens this to in-vivo measured values.
        # The kwargs are retained in the signature for backward compat with
        # callers but their numeric value is OVERRIDDEN here.
        _ = conduction_delay_a2msi  # kwarg silenced
        _ = conduction_delay_v2msi  # kwarg silenced
        self.conduction_delay_a2msi = 250   # 25.0 ms auditory (task#47 restore pre-f92d04a; cat-SC defensible per #45)
        self.conduction_delay_v2msi = int(os.environ.get("CONDUCTION_DELAY_V2MSI", "400"))   # 40.0ms default (V slower => V-leading TBW). LATENCY-FIX V-advance lever: reduce toward a2msi(250) so coincident V summates with A's sub-threshold rise (env-exposed; default 400 == byte-identical; load_ckpt restore wins post-load)
        self.conduction_delay_msi2out = conduction_delay_msi2out
        # delay-fix (recurrent MSI->MSI transmission delay): 100 substeps = 10.0 ms
        # = exactly 1 external step at n_substeps=100. Gives the recurrent synapse a
        # conduction delay so pre LEADS post, breaking the zero-delay antisymmetric-by-
        # construction LTP=LTD cancellation that froze W_MSI_exc at init (debugger H1).
        self.conduction_delay_msi_rec = 100

        # task #192 Phase A: REMOVED conduction_delay_inA_inh / _inV_inh
        # (paired with the direct FF-inh shortcut rip in __init__ above).

        # unimodal->MSI_inh excit:
        # task #192 Phase C: 4 ms (same as exc; one EPSC leg onto interneuron).
        self.conduction_delay_a2msi_inh = 270   # = a2msi + 20 (2.0 ms exc leg onto interneuron)
        self.conduction_delay_v2msi_inh = self.conduction_delay_v2msi + 20   # = v2msi + 20 (2.0ms disynaptic exc leg onto interneuron); TRACKS the V-advance lever to preserve the V exc/inh disynaptic offset

        # MSI_inh->MSI_exc
        # task #192 Phase C: 5.2 ms over the exc->inh leg, totalling
        # disynaptic ~9.2 ms (Whyland-Bickford 2018 IPSC).
        self.conduction_delay_msi_inh2exc = 52   # task TBW-delay-fix: local disynaptic IPSC ~5.2ms (Whyland-Bickford 2018); MEASUREMENT build matching the delay52 ckpt (was 450=45ms)

        # task #27: physical-time conduction delays (ms). Captured at construction
        # so the physical duration is dt-invariant. Used when self.dt_correct_nmda
        # is True; buffer sizing and access then compute substep counts as
        # int(round(*_ms / self.dt)).
        self.conduction_delay_a2msi_ms       = self.conduction_delay_a2msi       * self.dt
        self.conduction_delay_v2msi_ms       = self.conduction_delay_v2msi       * self.dt
        # task #192 Phase A: REMOVED conduction_delay_inA_inh_ms / _inV_inh_ms
        # (paired with the direct FF-inh shortcut rip).
        self.conduction_delay_a2msi_inh_ms   = self.conduction_delay_a2msi_inh   * self.dt
        self.conduction_delay_v2msi_inh_ms   = self.conduction_delay_v2msi_inh   * self.dt
        self.conduction_delay_msi_inh2exc_ms = self.conduction_delay_msi_inh2exc * self.dt
        self.conduction_delay_msi2out_ms     = self.conduction_delay_msi2out     * self.dt
        self.conduction_delay_msi_rec_ms     = self.conduction_delay_msi_rec     * self.dt

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
        self.tau_nmda_v = self.tau_nmda   # latency #153: V-specific FF-NMDA decay (default == tau_nmda; finalized in loader from env TAU_NMDA_V)
        self.tau_nmda_inh = 45.0  # task#14 route-c TBW fix: GluN2A-fast PV value (adult fast-spiking PV is GluN2A-dominated ~30-45ms). Was 25.0 (route-C TBW-tune); 90ms is a GluN2B miscite from rat CA1.
        self.nmda_alpha = 0.1
        self.mg_k = float(os.environ.get("MG_K", 0.062))                               # C1: NMDA Mg-block gate slope (env-config; default == original 0.062)
        self.Erev_nmda = 10.0
        self.tau_nmdaVolt = 200.0
        self.v_nmda_rest = -65.0
        self.nmda_vrest_offset = 7.0
        self.mg_vhalf = float(os.environ.get("MG_VHALF", -35.0))                       # P5: NMDA Mg half-relief V (env-config; default == original -35.0)
        self.mg_vhalf_inh = float(os.environ.get("MG_VHALF_INH", self.mg_vhalf))        # inh-gate DECOUPLE (debugger q5_inhvhalf): separate INH Mg half-relief V; default == exc mg_vhalf => byte-identical shared gate

        self.dend_coupling_alpha = float(os.environ.get("DEND_COUPLING_ALPHA", 0.1))    # P5: v_dend->Mg coupling rate (env-config; default == original 0.1)
        self.v_dend_A = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.v_dend_V = torch.full((self.batch_size, self.n), self.cM, device=self.device)

        # NMDA gating state (MSI excit)
        self.nmda_m = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        # latency #153: V-specific FF-NMDA gating pool (split-enabler; inert/zeros at tau_nmda_v==tau_nmda).
        self.nmda_m_v = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        # recurrent MSI->MSI NMDA gating state (Increment 1; inert at g_rec=0).
        self.nmda_m_rec = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.v_nmda = torch.full((self.batch_size, self.n), self.v_nmda_rest, dtype=torch.float32, device=self.device)

        self.v_dend_inhA = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.v_dend_inhV = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.nmda_m_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self.v_nmda_inh = torch.full((self.batch_size, self.n_inh), self.v_nmda_rest, dtype=torch.float32,
                                     device=self.device)

        ################################################################
        # Short-term depression (Tsodyks-Markram) for AMPA
        ################################################################
        # task#104 (decouple retrain): TM short-term-depression knobs, env-exposed (defaults byte-identical).
        # u_stp_a/u_stp_v = the afferent (A/V) utilization U; reset_state re-applies these EVERY trial (the dead
        # build-path fill_(0.7) is overwritten before any dynamics), so this scalar IS the operative utilization.
        # Raising U deepens the per-spike depression -> the 1st event depletes resources -> the late/2nd-event NMDA
        # bridge is suppressed (sparing the 1st-event coincidence peak). nmda_std_scale = the NMDA-afferent STD DEPTH
        # (forward scale_F: gate = 1 - scale_F*(1-R)); 0 = no NMDA STD, 1 = full (=AMPA). All defaults == current.
        self.u_stp_a = float(os.environ.get("U_STP_A", 0.2))
        self.u_stp_v = float(os.environ.get("U_STP_V", 0.2))
        self.nmda_std_scale = float(os.environ.get("NMDA_STD_SCALE", 0.8))
        # latency #148: V->MSI NMDA forward-conductance scale (V-specific receptor lever for onset-latency
        # facilitation). Strengthens the slow, voltage-dependent V NMDA to build a sub-threshold plateau "foot"
        # that gives summation ROOM for the #137 regenerative leap. Default 1.0 == byte-identical. Applied ONLY
        # at the V-NMDA forward conductance (NOT the trained weight or its plasticity); A->MSI + recurrent NMDA
        # untouched, and gNMDA stays global/unchanged. NEVER touches conduction delays.
        self.v2msi_nmda_scale = float(os.environ.get("V2MSI_NMDA_SCALE", 1.0))
        self.R_a = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_a = torch.full((self.batch_size, self.n), self.u_stp_a, device=self.device)
        self.R_v = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_v = torch.full((self.batch_size, self.n), self.u_stp_v, device=self.device)

        self.R_a_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_a_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)
        self.R_v_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_v_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)

        # task#385 SOM/Martinotti facilitation-in-time on the interneuron->exc GABA edge (debugger #384 spec).
        # Facilitating presynaptic utilization F_gaba: near-ZERO-early, GROWS-late (~100-300ms) -> vetoes LATE
        # cross-modal fusion (trims the TBW tail) while sparing offset~=0 (keeps dRACE/rate). Sign-flip of the
        # R_a_inh depression coded above; shape (B, n_inh) mirrors u_v_inh. All-default => OFF => byte-identical
        # (rel_sMi=new_sMi, substep block skipped, torch.full draws no RNG so no downstream init shifts).
        self.facil_gaba_on  = int(os.environ.get("FACIL_GABA_ON", "0"))
        self.tau_facil_gaba = float(os.environ.get("TAU_FACIL_GABA", "200.0"))
        self.U_facil_gaba   = float(os.environ.get("U_FACIL_GABA", "0.1"))
        self.F_gaba = torch.full((self.batch_size, self.n_inh), self.U_facil_gaba, device=self.device)

        self.tau_rec = float(os.environ.get("TAU_REC", 400.0))
        self.tau_fac = 20.0

        # --- debug counters -------------------------------------------------
        self._dbg_spk_A = 0.0  # accumulated spikes in layer A
        self._dbg_spk_V = 0.0  # accumulated spikes in layer V
        self._dbg_spk_MSI = 0.0  # accumulated spikes in MSI excit
        self._dbg_steps = 0  # how many external frames have been seen

        # --- task#21 per-epoch logging panel (read-only instrumentation) ----
        # All default-OFF / None so the panel is a strict no-op unless
        # run_training explicitly enables it (proven trajectory-neutral via the
        # §7 bit-identity smoke gate). _dbg_spk_Mi mirrors _dbg_spk_MSI for the
        # PV/FS (interneuron) rate (S1), normalised by n_inh (NOT n).
        self._panel_enabled = False
        self._panel_seed = None
        self._panel_rows = []
        self._panel_inh_accum = None   # E2: INH plateau accumulator (battery-only)
        self._panel_dW_accum = None    # E3: FF dW flux accumulator (Tap-A)
        self._last_rates_hz = None     # E1: rates stashed by print_epoch_spike_summary
        self._dbg_spk_Mi = 0.0         # E1/S1: accumulated spikes in MSI inhibitory

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
        # task #192 Phase A: STRIPPED eta_H / eta_AH (Mexican-hat learning rates).
        # These were orphan attributes (defined but never consumed in any update
        # loop). Mexican-hat KERNEL SHAPE is retained below as a fixed geometry.

        self.register_buffer("near_mask", torch.exp(-(dist / self.R_near) ** 2))
        self.register_buffer("far_mask", 1.0 - self.near_mask)

        # initialise learnable surround-inhibition matrix (non-negative)
        init_W = self.near_mask.clone()
        self.W_MSI_inh = nn.Parameter(Positive()(init_W),
                                      requires_grad=False)

        self.register_buffer("W_MSI_inh_init", init_W.clone())

        # task #132: bake the gated intermediate-surround into the (non-plastic) W_MSI_inh once at init.
        # budget==0.0 -> no-op -> byte-identical to the pre-knob build. budget!=0 reproduces the #131
        # debugger operator EXACTLY: ring = exp(-((dist_mask - d0_idx)/w_idx)^2), diagonal zeroed, scaled
        # so ring.sum() = budget × W_MSI_inh.sum(), then ADDED on top of the intact near-field. deg/idx via
        # the net's own geometry (space_size-1)/(n-1) (= 1.0°/idx at n=180, space_size=180 -> d0=15° = 15 idx).
        if self.surround_budget != 0.0:
            with torch.no_grad():
                deg_per_idx = (float(self.space_size) - 1.0) / (float(self.n) - 1.0)
                d0_idx = self.surround_d0_deg / deg_per_idx
                ring = torch.exp(-((self.dist_mask - d0_idx) / self.surround_w_idx) ** 2)
                ring.fill_diagonal_(0.0)
                ring *= (self.surround_budget * float(self.W_MSI_inh.sum()) / float(ring.sum()))
                self.W_MSI_inh.data.add_(ring)

        # ── recurrent MSI->MSI excitation — Increment 1 (INERT at g_rec=0) ──
        # Excitatory counterpart of the Mexican-hat surround inhibition above:
        # a SHORT-RANGE local Gaussian over the SAME MSI-sheet geometry
        # (self.dist_mask), driven by MSI spikes, carrying co-localised
        # AMPA+NMDA into the fast excitatory channel I_M (see forward pass).
        # The whole contribution is scaled by g_rec; default 0.0 => the pathway
        # is inert and inference is bit-identical to the pre-recurrence build.
        # No plasticity yet. Built deterministically (no RNG) so adding it does
        # NOT perturb the init RNG stream. Diagonal zeroed => no self-autapse.
        # R_rec / rec_*_frac are activation-increment tunables (no effect now).
        self.g_rec = 0.0           # recurrent excitation strength (Increment 1: OFF)
        self.R_rec = 2.0           # recurrent radius (neurons); narrower than R_near
        self.rec_ampa_frac = 0.25  # AMPA:NMDA split mirrors FF->MSI 0.25/0.75 renorm
        self.rec_nmda_frac = 0.75
        with torch.no_grad():
            _rec_kernel = torch.exp(-(self.dist_mask / self.R_rec) ** 2)
            _rec_kernel.fill_diagonal_(0.0)
            _W_rec_init = Positive()(_rec_kernel)
        self.W_MSI_exc = nn.Parameter(_W_rec_init, requires_grad=False)
        self.register_buffer("W_MSI_exc_init", _W_rec_init.clone())

        self.g_GABA = 2  # was 3 – stronger Mexican-hat inhibition
        # task #192 Phase A: g_FFinh retained as ORPHAN attribute only.
        # The direct FF-inh injection it scaled was removed (see __init__
        # rip block above for W_inA_inh/W_inV_inh + update_all_layers_batch
        # rip of the I_M.sub_(g_FFinh * ...) site). SBW/TBW probes still
        # save/restore this attribute, so we keep it defined (= 0.0) to
        # avoid AttributeError; assignments to it have no functional effect.
        self.g_FFinh = 0.0
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
        # task #188 (simple-pathway §4.1): NO-OP stub — iSTDP removed; method
        # retained for backward compat with train_and_save callers.
        self.allow_inhib_plasticity = enable  # orphan flag, not consulted anywhere

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
            # task #192 Phase A: REMOVED W_inA_inh / W_inV_inh zero calls
            # (those attributes no longer exist; direct FF-inh shortcut rip'd).
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
        # task #192 Phase A: REMOVED W_inA_inh / W_inV_inh prints (attributes
        # no longer exist; direct FF-inh shortcut rip'd).
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
    def _ensure_delay_buffer(self, attr: str, *, delay: int, width: int, extra: int = 0) -> None:
        """Allocate or re-allocate a single GPU ring buffer.

        `extra` (task #327): headroom substeps appended to the ring length so a
        per-element read offset (delay_base ± afferent-arrival jitter) stays in-buffer.
        extra==0 (default) => length==delay => byte-identical to pristine.
        """
        buf_len = max(int(delay) + int(extra), 1)
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
        # task #192 Phase A: REMOVED d_inA_inh / d_inV_inh buffer allocation
        # (paired with the direct FF-inh shortcut rip).
        if use_ms:
            d_a2msi       = self._delay_substeps_from_ms(self.conduction_delay_a2msi_ms)
            d_v2msi       = self._delay_substeps_from_ms(self.conduction_delay_v2msi_ms)
            d_a2msi_inh   = self._delay_substeps_from_ms(self.conduction_delay_a2msi_inh_ms)
            d_v2msi_inh   = self._delay_substeps_from_ms(self.conduction_delay_v2msi_inh_ms)
            d_msi_inh2exc = self._delay_substeps_from_ms(self.conduction_delay_msi_inh2exc_ms)
            d_msi2out     = self._delay_substeps_from_ms(self.conduction_delay_msi2out_ms)
            d_msi_rec     = self._delay_substeps_from_ms(self.conduction_delay_msi_rec_ms)
        else:
            d_a2msi       = self.conduction_delay_a2msi
            d_v2msi       = self.conduction_delay_v2msi
            d_a2msi_inh   = self.conduction_delay_a2msi_inh
            d_v2msi_inh   = self.conduction_delay_v2msi_inh
            d_msi_inh2exc = self.conduction_delay_msi_inh2exc
            d_msi2out     = self.conduction_delay_msi2out
            d_msi_rec     = self.conduction_delay_msi_rec
        # task #327 — afferent-arrival jitter headroom on the A/V forward ring buffers ONLY.
        # margin = ceil(5σ) substeps, capped at delay_base-1 (keeps symmetric ±margin clip mean-preserving
        # AND effective delay ≥ 1). OFF (afferent_jitter_ms==0) => margin 0 => length==delay => byte-identical.
        _aff_ms = float(getattr(self, 'afferent_jitter_ms', 0.0))
        if _aff_ms > 0.0:
            _aff_sig = _aff_ms / float(self.dt)
            _m_a = int(min(int(np.ceil(5.0 * _aff_sig)), int(d_a2msi) - 1))
            _m_v = int(min(int(np.ceil(5.0 * _aff_sig)), int(d_v2msi) - 1))
        else:
            _m_a = _m_v = 0
        self._aff_delay_base_a, self._aff_delay_base_v = int(d_a2msi), int(d_v2msi)
        self._aff_margin_a, self._aff_margin_v = _m_a, _m_v
        self._ensure_delay_buffer("buffer_a2msi",       delay=d_a2msi,       width=self.n, extra=_m_a)
        self._ensure_delay_buffer("buffer_v2msi",       delay=d_v2msi,       width=self.n, extra=_m_v)
        self._ensure_delay_buffer("buffer_a2msi_inh",   delay=d_a2msi_inh,   width=self.n)
        self._ensure_delay_buffer("buffer_v2msi_inh",   delay=d_v2msi_inh,   width=self.n)
        self._ensure_delay_buffer("buffer_msi_inh2exc", delay=d_msi_inh2exc, width=self.n_inh)
        self._ensure_delay_buffer("buffer_msi2out",     delay=d_msi2out,     width=self.n)
        self._ensure_delay_buffer("buffer_msi_rec",     delay=d_msi_rec,     width=self.n)

    def _p_add(self, attr: str, dW: torch.Tensor,
               eps: float = 1e-9,
               rel_clip: float = 0.25,
               abs_cap: float = 50.0,
               abs_step: float = 0.0):
        """Clipped additive update for direct weight parameters."""
        with torch.no_grad():
            W = getattr(self, attr)
            step_limit = (rel_clip * W.abs().clamp_min(eps)).clamp_min(abs_step)  # task#155: abs_step floor breaks compound-growth bottleneck for iSTDP
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

    def _get_cached_gaussian_kernel_sharpened(self, sigma: float) -> torch.Tensor:
        """task #120 — unimodal-anchor Gaussian, optionally afferent-SHARPENED.

        f = self.afferent_sharpen_factor (DEFAULT 1.0). f == 1.0 returns the EXACT
        _get_cached_gaussian_kernel(sigma) (same cached tensor) -> byte-identical OFF path.
        f < 1.0: narrow sigma -> sigma*f AND scale by 1/f so each column-sum (total afferent
        charge per unit) is conserved (narrower-but-taller; Shah&Crair area×peak invariant, #118 §3).
        Scoped to the unimodal input anchor (W_inA/W_inV) only — apply_topographic_anchor_msi/_inh
        keep calling _get_cached_gaussian_kernel unchanged.
        """
        f = getattr(self, 'afferent_sharpen_factor', 1.0)
        if f == 1.0:
            return self._get_cached_gaussian_kernel(sigma)
        key = ("gauss_sharp", sigma, f)
        if key not in self._kernel_cache:
            idx = torch.arange(self.n, device=self.device, dtype=torch.float32)
            self._kernel_cache[key] = (1.0 / f) * torch.exp(
                -0.5 * ((idx[:, None] - idx[None, :]) / (sigma * f)) ** 2
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

    def _get_cached_cross_gaussian_kernel(self, sigma: float,
                                         n_post: int, n_pre: int) -> torch.Tensor:
        """Return (n_post, n_pre) cross-population Gaussian kernel.

        task #192 Phase D: positions are spread uniformly over the [0, n_pre-1]
        axis (so n_post evenly-spaced postsynaptic neurons project
        topographically onto n_pre presynaptic neurons). Used by
        apply_topographic_anchor_msi_inh for the (n_inh, n) input->interneuron
        Hebbian+Oja update.
        """
        key = ("cross_gauss", sigma, n_post, n_pre)
        if key not in self._kernel_cache:
            denom = float(max(n_post - 1, 1))
            post_pos = (torch.arange(n_post, device=self.device,
                                     dtype=torch.float32)
                        * float(n_pre - 1) / denom)
            pre_pos = torch.arange(n_pre, device=self.device,
                                   dtype=torch.float32)
            d = post_pos[:, None] - pre_pos[None, :]
            self._kernel_cache[key] = torch.exp(-0.5 * (d / sigma) ** 2)
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
        # task #192 Phase A: removed "W_inA_inh", "W_inV_inh" from list
        # (those attributes no longer exist; direct FF-inh shortcut rip'd).
        nonnegative_attrs = (
            "W_msiInh2Exc_GABA",
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

        # task #192 Phase A: drop now-removed weight attrs from legacy ckpts
        # so strict=True load doesn't fail on unexpected keys.
        for stale_attr in ("W_inA_inh", "W_inV_inh"):
            translated.pop(stale_attr, None)
            translated.pop(f"parametrizations.{stale_attr}.original", None)

        return translated

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        translated = dict(self._translate_legacy_state_dict(state_dict))
        # Increment 1 (recurrent MSI->MSI excitation): pre-recurrence ckpts have
        # no W_MSI_exc / W_MSI_exc_init. Inject the freshly-built deterministic
        # values so a strict load of a legacy ckpt succeeds, WITHOUT weakening
        # strict checking for any other key. New (post-recurrence) ckpts already
        # carry these keys, so the injection is skipped for them.
        _own = self.state_dict()
        for _k in ("W_MSI_exc", "W_MSI_exc_init"):
            if _k in _own and _k not in translated:
                translated[_k] = _own[_k].clone()
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
        Binary-search `self.input_scaling` so that a *single-modality* pulse
        elicits `target_MSI_spikes` +/- `tol`.

        task #192 Phase A: removed the trailing g_FFinh sweep that used to
        follow this gain search. The direct FF-inh injection it tuned was
        rip'd (no W_inA_inh / W_inV_inh anymore), so sweeping g_FFinh now
        has no functional effect on response. Input-gain calibration alone
        is retained.
        """
        print(f"[CAL] calibrating input_scaling  "
              f"target_MSI_spikes={target_MSI_spikes}  tol={tol}  "
              f"max_iter={max_iter}  high_bound={high_bound}  "
              f"probe_frames={probe_frames}")

        lo, hi = 0.0, high_bound
        best_gain, best_err = self.input_scaling, float("inf")

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

    def _probe_msi_bimodal_spike_sum(
            self,
            *,
            stim_peak: float = 1.0,
            sigma_in: float = 5.0,
            ctr: int = 90,
            probe_frames: int = 15,
    ) -> float:
        """
        Bimodal-Gaussian variant of ``_probe_spike_sum`` for use by the
        Phase E iSTDP calibrator. The single-neuron audio pulse used by
        the unimodal probe is too weak to drive the untrained MSI_exc
        layer with the new biology-anchored inhibition (Phases A-D);
        a coincident A+V Gaussian (matching real stimuli) is needed so
        the binary search has a non-zero spike signal.

        Returns the *integrated* MSI_exc spike count over ``probe_frames``
        outer-loop frames, with plasticity disabled (so the probe does
        not perturb weights). Restores ``plasticity_enabled`` on exit.
        """
        prev_plasticity = getattr(self, 'plasticity_enabled', True)
        self.plasticity_enabled = False

        self.reset_state(batch_size=1)
        xs = torch.arange(self.n, device=self.device).float()
        gauss = torch.exp(-0.5 * ((xs - float(ctr)) / float(sigma_in)) ** 2)
        gauss = (stim_peak * gauss).unsqueeze(0)   # (1, n)

        tot = 0.0
        try:
            for _ in range(probe_frames):
                *_, sSum = self.update_all_layers_batch(
                    gauss, gauss,                  # bimodal coincident
                    return_spike_sum=True,
                )
                tot += sSum.sum().item()
        finally:
            self.plasticity_enabled = prev_plasticity
        return tot

    def auto_calibrate_W_msiInh2Exc_GABA(
            self,
            target_MSI_spikes: int = 400,
            tol: int = 40,
            max_iter: int = 12,
            scale_lo: float = 0.0,
            scale_hi: float = 200.0,
            probe_frames: int = 15,
            probe_stim_peak: float = 1.0,
            probe_sigma_in: float = 5.0,
    ):
        """
        task #192 Phase E (sci_inhibition_functional_role.md §5.4b):
        Binary-search a multiplicative scale on ``W_msiInh2Exc_GABA`` so a
        bimodal Gaussian probe yields ``target_MSI_spikes +/- tol`` total
        MSI spikes, anchoring the initial MSI rate to the biology target
        (~50 Hz peak; spec §5.4b) before iSTDP refines weights during
        training.

        Larger scale -> stronger disynaptic GABA -> fewer MSI spikes.
        Smaller scale -> weaker disynaptic GABA -> more MSI spikes.

        The baseline weights captured in ``W_msiInh2Exc_GABA_init`` are
        used as the unscaled reference so repeated calls do not drift.
        The result is clamped to ``[0, W_gaba_clamp]`` per spec.

        NOTE: This requires MSI_exc to be excitable from raw bimodal
        input. With the new biology-anchored Phases A-D, a fresh
        untrained network may not produce MSI_exc spikes even under
        bimodal coincident drive (the W_a2msi / W_v2msi weights need
        unimodal training first). In that case the calibrator will
        converge to scale=lower-bound (zero inhibition) which is still
        a defensible initialization (iSTDP grows W during training).
        """
        print(f"[CAL] calibrating W_msiInh2Exc_GABA  "
              f"target_MSI_spikes={target_MSI_spikes}  tol={tol}  "
              f"max_iter={max_iter}  "
              f"scale_lo={scale_lo:.2f} scale_hi={scale_hi:.2f}  "
              f"probe_frames={probe_frames}  "
              f"probe_stim_peak={probe_stim_peak:.2f}  "
              f"probe_sigma_in={probe_sigma_in:.2f}")

        # Reference baseline (the original sample from U[0, 0.002))
        W_init = self.W_msiInh2Exc_GABA_init.clone()

        lo, hi = float(scale_lo), float(scale_hi)
        best_scale, best_err = 1.0, float("inf")

        for _it in range(max_iter):
            scale = 0.5 * (lo + hi)
            with torch.no_grad():
                new_W = (W_init * scale).clamp_(min=0.0, max=self.W_gaba_clamp)
                self.W_msiInh2Exc_GABA.data.copy_(new_W)

            excit = self._probe_msi_bimodal_spike_sum(
                stim_peak=probe_stim_peak,
                sigma_in=probe_sigma_in,
                probe_frames=probe_frames,
            )
            err = abs(excit - target_MSI_spikes)

            print(f"    [CAL it={_it:02d}] lo={lo:8.2f} hi={hi:8.2f} "
                  f"scale={scale:8.2f}  spikes={excit:8.2f}  err={err:8.2f}")

            if err < best_err:
                best_scale, best_err = scale, err
            if err <= tol:
                break
            # More spikes than target -> need MORE inhibition -> raise scale
            if excit > target_MSI_spikes:
                lo = scale
            else:
                hi = scale

        # Apply best scale
        with torch.no_grad():
            new_W = (W_init * best_scale).clamp_(min=0.0, max=self.W_gaba_clamp)
            self.W_msiInh2Exc_GABA.data.copy_(new_W)
        converged = best_err <= tol
        print(f"[CAL] W_msiInh2Exc_GABA scale done: scale={best_scale:.2f}  "
              f"best_err={best_err:.2f}  converged={converged}  "
              f"W_sum={self.W_msiInh2Exc_GABA.sum().item():.4f}  "
              f"W_max={self.W_msiInh2Exc_GABA.max().item():.4f}")

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

    # ====================================================================
    # task#21 per-epoch logging panel (E4/E5) — READ-ONLY instrumentation.
    # Trajectory-neutral: the battery saves/restores RNG+step_counter+flags+
    # _ei_record (§5); plasticity is OFF; transient state is auto-wiped by the
    # next epoch's reset_state. Proven by the §7 bit-identity smoke gate.
    # ====================================================================
    @staticmethod
    def _sarle_bc(x):
        """Sarle's bimodality coefficient over a value vector (P6 fusion-shape).
        High (->1) = bimodal/box-like; low = graded/unimodal."""
        x = np.asarray(x, dtype=float); n = x.size
        if n < 4:
            return float('nan')
        s = x.std()
        if s <= 1e-12:
            return float('nan')
        z = (x - x.mean()) / s
        g1 = float(np.mean(z ** 3))           # skewness
        g2 = float(np.mean(z ** 4) - 3.0)      # excess kurtosis
        denom = g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        return (g1 ** 2 + 1.0) / denom if denom != 0 else float('nan')

    def _tbw_reduce(self, p, soa):
        """P1/P6 reductions from a P(fusion) curve `p` over ms-offsets `soa`."""
        out = {}
        i0 = int(np.argmin(np.abs(soa)))
        out['P1_P_at_0'] = float(p[i0])
        out['P1_P_at_m400'] = float(p[0])
        out['P1_P_at_p400'] = float(p[-1])
        above = p >= 0.5
        best = None; lo = None
        for i, a in enumerate(above):
            if a and lo is None:
                lo = i
            if (not a or i == len(above) - 1) and lo is not None:
                hi = i if a else i - 1
                if best is None or (hi - lo) > (best[1] - best[0]):
                    best = (lo, hi)
                lo = None
        if best is None:
            out['P1_width_ms'] = 0.0
            out['P1_lo_ms'] = float('nan'); out['P1_hi_ms'] = float('nan')
            out['P1_midpoint_ms'] = float('nan')
        else:
            lo_ms, hi_ms = float(soa[best[0]]), float(soa[best[1]])
            out['P1_width_ms'] = hi_ms - lo_ms
            out['P1_lo_ms'] = lo_ms; out['P1_hi_ms'] = hi_ms
            out['P1_midpoint_ms'] = 0.5 * (lo_ms + hi_ms)
        out['P6_n_graded'] = int(np.sum((p > 0.02) & (p < 0.98)))
        out['P6_n_offsets'] = int(p.size)
        out['P6_bc'] = float(self._sarle_bc(p))
        return out

    def _panel_ff_sigma(self, path="A2MSI"):
        """Mean effective spatial σ (neurons) of FF weight rows (S5; vectorised
        equivalent of feedforward_row_stats, which only prints)."""
        if path == "A2MSI":
            W = (self.W_a2msi_AMPA + self.W_a2msi_NMDA).detach().cpu()
        else:
            W = (self.W_v2msi_AMPA + self.W_v2msi_NMDA).detach().cpu()
        n = W.shape[1]
        xs = torch.arange(n, dtype=torch.float32)
        rs = W.sum(1)
        keep = rs > 0
        if int(keep.sum()) == 0:
            return float('nan')
        Wk = W[keep]; rk = rs[keep]
        mu = (Wk * xs).sum(1) / rk
        var = (Wk * (xs - mu.unsqueeze(1)) ** 2).sum(1) / rk
        return float(var.clamp_min(0).sqrt().mean().item())

    def _panel_single_volley(self, B=8, T_frames=120):
        """ONE coincident A+V volley (offset=0, noise off) — same path as the
        canonical TBW probe (mirrors dbg18 build_single_volley)."""
        loc = self.space_size // 2
        ls, msq = generate_two_event_offset_seq(loc=int(loc), T=T_frames, D=5,
                                                offset=0, space_size=self.space_size)
        xA, xV, mask = generate_av_batch_tensor(
            [ls] * B, [msq] * B, [False] * B, n=self.n, space_size=self.space_size,
            sigma_in=self.sigma_in, noise_std=0.0, device=self.device,
            max_len=T_frames, stimulus_intensity=1.0)
        return xA, xV, mask

    def _panel_weights_readout(self):
        """Direct-read weight reductions: P3 (iSTDP GABA), P4 (FF->MSI exc),
        H2 (frozen FF->INH). No forward pass, no RNG."""
        r = {}
        g = self.W_msiInh2Exc_GABA.detach()
        if getattr(self, '_panel_gaba_init', None) is None:
            self._panel_gaba_init = float(g.mean().item())
        r['P3_GABA_mean'] = float(g.mean().item())
        r['P3_GABA_std'] = float(g.std().item())
        r['P3_GABA_max'] = float(g.max().item())
        r['P3_GABA_clampfrac'] = float((g >= 0.999 * self.W_gaba_clamp).float().mean().item())
        r['P3_GABA_drift_vs_init'] = float(g.mean().item()) - self._panel_gaba_init
        sb = getattr(self, '_softbound_wmax', {}) or {}
        for nm in ('W_a2msi_AMPA', 'W_a2msi_NMDA', 'W_v2msi_AMPA', 'W_v2msi_NMDA'):
            w = getattr(self, nm).detach()
            r['P4_%s_mean' % nm] = float(w.mean().item())
            r['P4_%s_max' % nm] = float(w.max().item())
            r['P4_%s_rowsigma' % nm] = float(w.mean(dim=1).std().item())
            wmax = sb.get(nm)
            r['P4_%s_clampfrac' % nm] = (float((w >= 0.999 * wmax).float().mean().item())
                                         if wmax else float('nan'))

        def _mean(nm):
            return float(getattr(self, nm).detach().mean().item())
        r['P4_a_AMPA2NMDA'] = _mean('W_a2msi_AMPA') / (_mean('W_a2msi_NMDA') + 1e-12)
        r['P4_v_AMPA2NMDA'] = _mean('W_v2msi_AMPA') / (_mean('W_v2msi_NMDA') + 1e-12)
        for nm in ('W_a2msiInh_AMPA', 'W_a2msiInh_NMDA', 'W_v2msiInh_AMPA', 'W_v2msiInh_NMDA'):
            r['H2_%s_mean' % nm] = float(getattr(self, nm).detach().mean().item())
        return r

    def _panel_health_readout(self):
        """H1 numerical health + H3 E:I count (sanity; cheap, every epoch)."""
        r = {}
        vm = getattr(self, 'v_msi', None)
        r['H1_vmsi_min'] = float(vm.min().item()) if torch.is_tensor(vm) else float('nan')
        r['H1_vmsi_max'] = float(vm.max().item()) if torch.is_tensor(vm) else float('nan')
        vmi = getattr(self, 'v_msi_inh', None)
        r['H1_vmsiinh_min'] = float(vmi.min().item()) if torch.is_tensor(vmi) else float('nan')
        r['H1_vmsiinh_max'] = float(vmi.max().item()) if torch.is_tensor(vmi) else float('nan')
        nan_ct = 0
        for nm in ('W_inA', 'W_inV', 'W_a2msi_AMPA', 'W_a2msi_NMDA', 'W_v2msi_AMPA',
                   'W_v2msi_NMDA', 'W_msiInh2Exc_GABA', 'W_a2msiInh_AMPA', 'W_v2msiInh_AMPA',
                   'v_msi', 'v_msi_inh', 'post_rate_avg'):
            t = getattr(self, nm, None)
            if torch.is_tensor(t):
                nan_ct += int(torch.isnan(t).sum().item())
        r['H1_nan_count'] = int(nan_ct)
        r['H3_n'] = int(self.n)
        r['H3_n_inh'] = int(self.n_inh)
        return r

    @torch.no_grad()
    def _panel_battery(self, seed, full=False):
        """Tap-B controlled probe (plasticity-OFF, fixed seed). Returns a flat
        dict of reductions. SAVE/RESTORE RNG+step_counter+flags+_ei_record so the
        whole battery is a trajectory no-op (§5)."""
        from TBW_test import compute_tbw_temporal_fusion_persep
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        rng_np = np.random.get_state()  # DBG22-FIX save numpy RNG
        np.random.seed(int(seed))
        saved = dict(step=int(self.step_counter), plast=self.plasticity_enabled,
                     probe=self.enable_probe, ei=self._ei_record)
        self.plasticity_enabled = False
        self.enable_probe = False
        self._ei_record = None
        self._panel_inh_accum = None
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        out = {}
        try:
            # (a) TBW (P1/P6) — cadence-selected grid (1 frame = 10 ms)
            if full:
                offs = list(range(-40, 41, 2)); ntr = 50           # 41-pt full
            else:
                offs = sorted(list(range(-20, 21, 2)) + [-40, -32, -24, 24, 32, 40])
                ntr = 20                                           # 27-pt reduced
            p_fusion, _af = compute_tbw_temporal_fusion_persep(
                self, offs, n_trials=ntr, T=60, D=5, sigma=2.0, valley_threshold=0.4,
                min_peak_height=0.2, min_peak_separation=3, min_total=10.0)
            p = np.asarray([float(x) for x in p_fusion], dtype=float)
            soa = np.asarray([o * 10.0 for o in offs], dtype=float)
            out.update(self._tbw_reduce(p, soa))

            # (b,d,e,S5) ONE coincident single-volley: E/I + plateau + rate + spatial-FWHM
            self.reset_state(8)
            self._panel_inh_accum = {k: [] for k in
                ('mg_iA', 'mg_iV', 'mg_iA_on', 'mg_iV_on', 'v_dend_inhA', 'v_dend_inhV')}
            self.start_ei_recording()
            xA, xV, mask = self._panel_single_volley(B=8, T_frames=120)
            for t in range(xA.shape[1]):
                self.update_all_layers_batch(xA[:, t], xV[:, t], mask[:, t], return_spike_sum=True)
            ei = self.stop_ei_recording()
            acc = self._panel_inh_accum; self._panel_inh_accum = None

            def _m(k):
                v = ei.get(k)
                return float(np.mean(v)) if (v is not None and len(v)) else float('nan')
            QE, QI = _m('Q_E'), _m('Q_I')
            out['P2B_Q_E'] = QE; out['P2B_Q_I'] = QI
            out['P2B_EI_ratio'] = (QE / QI) if (QI and not np.isnan(QI)) else float('nan')
            out['P2B_AMPA'] = _m('AMPA'); out['P2B_NMDA'] = _m('NMDA')
            out['P2B_RecurInh'] = _m('RecurInh'); out['P2B_LatInh'] = _m('LatInh')
            out['P2B_I_GABA'] = out['P2B_RecurInh'] + out['P2B_LatInh']
            # plateau (S2): NMDA-plateau duration from per-substep mg-on series
            res_ms = float(self.dt)
            mga_on = np.asarray(acc['mg_iA_on'], dtype=float)
            mgv_on = np.asarray(acc['mg_iV_on'], dtype=float)
            on = (mga_on >= 0.5) | (mgv_on >= 0.5)
            idx = np.where(on)[0]
            out['S2_plateau_dur_ms'] = float((idx[-1] - idx[0] + 1) * res_ms) if idx.size else 0.0
            out['S2_mg_iA_frac'] = float(np.mean(mga_on)) if mga_on.size else float('nan')
            out['S2_mg_iV_frac'] = float(np.mean(mgv_on)) if mgv_on.size else float('nan')
            out['S2_v_dend_inhA'] = float(np.mean(acc['v_dend_inhA'])) if acc['v_dend_inhA'] else float('nan')
            out['S2_v_dend_inhV'] = float(np.mean(acc['v_dend_inhV'])) if acc['v_dend_inhV'] else float('nan')
            # rate (P5-B) + sparseness (S7) from per-neuron post_rate_avg
            rate = self.post_rate_avg.mean(0).detach()
            hz = rate * (1000.0 / float(self.dt))
            out['P5_peak_hz'] = float(hz.max().item())
            n_act = int((hz > 1.0).sum().item())
            out['P5_active_mean_hz'] = float(hz[hz > 1.0].mean().item()) if n_act else 0.0
            out['P5_std_hz'] = float(hz.std().item())
            out['S7_n_active'] = n_act
            out['S7_pct_active'] = float((hz > 1.0).float().mean().item())
            # (S5) spatial FWHM of the MSI population response
            try:
                out['S5_msi_fwhm_deg'] = float(msi_pop_fwhm(rate, space_size=self.space_size))
            except Exception:
                out['S5_msi_fwhm_deg'] = float('nan')

            if full:
                out['S5_ff_sigma'] = self._panel_ff_sigma("A2MSI")
                # (f) MSE / multisensory enhancement (S6). self.evaluate_batch
                # (substrate @3409) is otherwise-uncalled code with a latent final_t
                # overflow (`non_blank[-1]+5` can exceed len(loc_seq) @~3489 ->
                # IndexError). S6/MSE is the LOWEST-priority, every-5, sheddable metric
                # (lead §0): degrade to NaN on ANY failure rather than crash the
                # otherwise-valid panel-ON run. Substrate evaluate_batch is NOT touched.
                try:
                    e_both = float(self.evaluate_batch(50, condition="both", batch_size=50))
                    e_a = float(self.evaluate_batch(50, condition="audio_only", batch_size=50))
                    e_v = float(self.evaluate_batch(50, condition="visual_only", batch_size=50))
                    u = min(e_a, e_v)
                    out['S6_err_both'] = e_both; out['S6_err_audio'] = e_a; out['S6_err_visual'] = e_v
                    out['S6_ME_pct'] = 100.0 * (e_both - u) / u if u > 1e-9 else float('nan')
                except Exception as _s6e:
                    out['S6_err_both'] = float('nan'); out['S6_err_audio'] = float('nan')
                    out['S6_err_visual'] = float('nan'); out['S6_ME_pct'] = float('nan')
                # (g) MSI_inh STP resources (S4)
                out['S4_R_a_inh'] = float(self.R_a_inh.mean().item())
                out['S4_R_v_inh'] = float(self.R_v_inh.mean().item())
        finally:
            self.plasticity_enabled = saved['plast']
            self.enable_probe = saved['probe']
            self._ei_record = saved['ei']
            self.step_counter = saved['step']
            self._panel_inh_accum = None
            np.random.set_state(rng_np)  # DBG22-FIX restore numpy RNG
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state_all(rng_cuda)
        return out

    def _epoch_panel_dump(self, epoch, seed):
        """Assemble + append ONE panel row for (epoch, seed). Tap-A scalars are
        read BEFORE the battery (which reset_states and would zero them)."""
        if not getattr(self, '_panel_enabled', False):
            return None
        row = {'epoch': int(epoch), 'seed': int(seed)}
        full = (int(epoch) % 5 == 0)
        # (1) Tap-A FIRST: P5/S1 rates (stashed by print_epoch_spike_summary)
        lr = self._last_rates_hz or {}
        row['P5_A_hz'] = float(lr.get('A', float('nan')))
        row['P5_V_hz'] = float(lr.get('V', float('nan')))
        row['P5_MSI_hz'] = float(lr.get('MSI', float('nan')))
        row['S1_MSIinh_hz'] = float(lr.get('MSI_inh', float('nan')))
        # P2-A in-training E/I: stop + reduce Tap-A _ei_record
        ei_keys = ['P2A_Q_E', 'P2A_Q_I', 'P2A_EI_ratio', 'P2A_AMPA', 'P2A_NMDA',
                   'P2A_FFInh', 'P2A_RecurInh', 'P2A_LatInh', 'P2A_I_GABA']
        if self._ei_record is not None:
            ei = self.stop_ei_recording()

            def _m(k):
                v = ei.get(k)
                return float(np.mean(v)) if (v is not None and len(v)) else float('nan')
            QE, QI = _m('Q_E'), _m('Q_I')
            row['P2A_Q_E'] = QE; row['P2A_Q_I'] = QI
            row['P2A_EI_ratio'] = (QE / QI) if (QI and not np.isnan(QI)) else float('nan')
            row['P2A_AMPA'] = _m('AMPA'); row['P2A_NMDA'] = _m('NMDA')
            row['P2A_FFInh'] = _m('FFInh'); row['P2A_RecurInh'] = _m('RecurInh')
            row['P2A_LatInh'] = _m('LatInh')
            row['P2A_I_GABA'] = row['P2A_RecurInh'] + row['P2A_LatInh']
        else:
            for k in ei_keys:
                row[k] = float('nan')
        # S3 dW flux: reduce + clear
        dW = self._panel_dW_accum or {}
        tot = 0.0
        for nm in ('W_inA', 'W_inV', 'W_a2msi_AMPA', 'W_v2msi_AMPA',
                   'W_a2msi_NMDA', 'W_v2msi_NMDA'):
            v = float(dW.get(nm, float('nan')))
            row['S3_dW_' + nm] = v
            if v == v:  # not NaN
                tot += v
        row['S3_dW_total'] = tot if dW else float('nan')
        self._panel_dW_accum = None
        # (2) direct-read weights (P3,P4,H2) + (3) health (H1,H3)
        row.update(self._panel_weights_readout())
        row.update(self._panel_health_readout())
        # (5) battery (Tap-B)
        row.update(self._panel_battery(seed, full=full))
        # (6) append
        self._panel_rows.append(row)
        return row

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
        self.msi_inh_refrac = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)

        # reset out
        self.v_out = torch.full((self.batch_size, self.n), self.cO, dtype=torch.float32, device=self.device)
        self.u_out = self.bO * self.v_out

        self.I_A = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_V = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_M = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        # task #192 Phase B: also reset I_M_gaba (slow-decaying GABA state).
        self.I_M_gaba = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_M_gaba_surr_sh = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._last_I_surr_shunt = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)  # task #55: shunt-aware E/I exposure; 0 until first substep / 0 when OFF
        self.I_M_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self.I_O = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_ampa_filtered = torch.zeros((self.batch_size, self.n), device=self.device)
        # task#130 Form-A: reset the dVm/dt low-pass trace per trial (runtime state, not a param; OFF => unused, byte-identical)
        self.dvdt_trace_msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)

        # task#114 STEP-1: reset substep first-spike probe per trial (clock from 0; _fs_probe_on persists)
        self._first_spike_substep = torch.full((self.batch_size,), -1, dtype=torch.long, device=self.device)
        self._fs_substep_clock = 0
        self._fs_dvdt_trace = torch.full((self.batch_size,), float('nan'), dtype=torch.float32, device=self.device)  # task#130 anchor: dV/dt trace at the crossing substep (probe+Form-A only)
        self._v_msi_substep_trace = []   # latency #152: clear substep max-v_msi trace per trial (mirrors _first_spike_substep; logging-only)
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
        # task #192 Phase E: reset iSTDP traces (interneuron -> MSI_exc edge).
        self.pre_trace_msiInh2Exc = torch.zeros(
            (self.batch_size, self.n_inh), dtype=torch.float32, device=self.device
        )
        self.post_trace_msiInh2Exc = torch.zeros(
            (self.batch_size, self.n), dtype=torch.float32, device=self.device
        )
        # Increment 3: re-zero recurrent MSI->MSI excitatory STDP traces.
        self.pre_trace_msi_rec = torch.zeros(
            (self.batch_size, self.n), dtype=torch.float32, device=self.device
        )
        self.post_trace_msi_rec = torch.zeros(
            (self.batch_size, self.n), dtype=torch.float32, device=self.device
        )
        # delay-fix: clear the delayed recurrent-STDP pre at each sequence start.
        self._prev_sMSI_rec = None
        self.ampa_m.zero_()  # clear low-pass AMPA state

        # clear conduction ring buffers
        self._reset_delay_buffers()

        # task #327 — per-TRIAL, per-CHANNEL INDEPENDENT afferent-arrival jitter (integer substep offsets
        # applied to the A/V ring-buffer READ). GATED: OFF => draw NOTHING (RNG stream untouched) => set
        # _aff_jitter_* None => forward takes the scalar buf[pos] path => BYTE-IDENTICAL. ON => two independent
        # (batch,) Gaussian draws (σ = afferent_jitter_ms/dt substeps), symmetric-clipped to ±margin
        # (mean-preserving); +offset = longer conduction delay (later arrival), -offset = earlier.
        if float(getattr(self, 'afferent_jitter_ms', 0.0)) > 0.0:
            _aff_sig = float(self.afferent_jitter_ms) / float(self.dt)
            self._aff_jitter_a = torch.round(
                torch.randn(self.batch_size, device=self.device) * _aff_sig
            ).long().clamp_(-int(self._aff_margin_a), int(self._aff_margin_a))
            self._aff_jitter_v = torch.round(
                torch.randn(self.batch_size, device=self.device) * _aff_sig
            ).long().clamp_(-int(self._aff_margin_v), int(self._aff_margin_v))
            self._aff_batch_arange = torch.arange(self.batch_size, device=self.device)
        else:
            self._aff_jitter_a = None
            self._aff_jitter_v = None

        if hasattr(self, "ampa_m"):
            self.ampa_m = torch.zeros((self.batch_size, self.n),
                                      dtype=torch.float32,
                                      device=self.device)

        # reset NMDA gating
        self.nmda_m = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        # latency #153: V-specific FF-NMDA gating pool (split-enabler; inert/zeros at tau_nmda_v==tau_nmda).
        self.nmda_m_v = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        # recurrent MSI->MSI NMDA gating state (Increment 1; inert at g_rec=0).
        self.nmda_m_rec = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.v_nmda = torch.full((self.batch_size, self.n), self.v_nmda_rest, dtype=torch.float32, device=self.device)
        self.nmda_m_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self.v_nmda_inh = torch.full((self.batch_size, self.n_inh), self.v_nmda_rest, dtype=torch.float32,
                                     device=self.device)

        self.v_dend_A = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.v_dend_V = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.v_dend_inhA = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.v_dend_inhV = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)

        # reset STP  (task#104: u_a/u_v re-applied from the env-exposed operative utilization; default 0.2 byte-identical)
        self.R_a = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_a = torch.full((self.batch_size, self.n), self.u_stp_a, device=self.device)
        self.R_v = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_v = torch.full((self.batch_size, self.n), self.u_stp_v, device=self.device)

        self.R_a_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_a_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)
        self.R_v_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_v_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)

        # task#385: reset facilitating utilization to baseline U each trial (per-trial transient, (B, n_inh)).
        self.F_gaba = torch.full((self.batch_size, self.n_inh), self.U_facil_gaba, device=self.device)

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
        self._dbg_spk_Mi = 0.0  # task#21/S1: PV/FS spike accumulator

        # task #185 (minimal-arch §5.1): REMOVED AGC gate-state reset
        # (_last_agc_fast_t, _last_agc_slow_t no longer exist).

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
        # task#21/S1: PV/FS (MSI inhibitory) rate — normalise by n_inh, NOT n
        rMi = self._dbg_spk_Mi / (self._dbg_steps * self.n_inh)

        # Convert to Hz
        ms_per_frame = self.n_substeps * self.dt
        hz_fact = 1000.0 / ms_per_frame
        rA_hz, rV_hz, rM_hz = (x * hz_fact for x in (rA, rV, rM))
        rMi_hz = rMi * hz_fact  # task#21/S1

        print(f"[rate] {tag:10s}"
              f"  A={rA_hz:6.2f} Hz"
              f"  V={rV_hz:6.2f} Hz"
              f"  MSI={rM_hz:6.2f} Hz")
              # task #188: REMOVED (target=rho0*hz_fact) — no iSTDP setpoint

        # task#21/E1: stash rates for the panel BEFORE the reset zeroes them (P5/S1)
        self._last_rates_hz = dict(A=rA_hz, V=rV_hz, MSI=rM_hz, MSI_inh=rMi_hz)

        # ready for next epoch
        self._dbg_spk_A = self._dbg_spk_V = self._dbg_spk_MSI = 0.0
        self._dbg_spk_Mi = 0.0
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
        nmda_decay_v = 1.0 - self.dt / self.tau_nmda_v   # latency #153: V-specific FF-NMDA decay (default == nmda_decay)
        nmda_decay_inh = 1.0 - self.dt / self.tau_nmda_inh  # task#51 route-c (interneuron NMDA decay)
        # task #192 Phase B: separate slow GABA decay (tau_gaba=50ms biology).
        gaba_decay = 1.0 - self.dt / self.tau_gaba
        # Per-step source-onto-decaying-state scale (task #16/#121/#123/#135).
        #   dt_linear_scale: linear `dt / 0.1` applied uniformly to AMPA and NMDA
        #     injection sites (and the AGC/FF-inh AMPA path). At canonical dt=0.1
        #     this is 1.0 (byte-identical to bare add → legacy ckpts produce
        #     paper biology unchanged). At other dt it scales per-substep
        #     injection so total per-ms injection is preserved → dt-invariant.
        #     Task #135 unified NMDA onto this same scaling (previously used
        #     exp-Euler Form 2 `1 - exp(-dt/tau_syn)` ≈ 0.0392 at dt=0.1, which
        #     broke paper output on legacy ckpts).
        dt_linear_scale = self.dt / 0.1
        input_step_scale = 1.0 / float(self.n_substeps)
        # task #188: REMOVED istdp_decay precomputation (no iSTDP rule).
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
        # task #192 Phase A: REMOVED W_inA_inh / W_inV_inh local aliases
        # (direct FF-inh shortcut rip'd).
        # task#102 (B): persistent forward-multiply on the FFI-recruitment weights. msiInh_input_gain (default 1.0 =>
        # byte-identical, 1.0*W is exact) scales the afferent->interneuron drive EVERY step on top of the trained
        # weights -> recruitment cannot wash out (unlike init-scaling). Scales AMPA + NMDA, A + V together, since these
        # forward-local aliases are the SOLE consumers (F.linear into the interneuron, below).
        _ffi_g = self.msiInh_input_gain
        W_a2msiInh_AMPA = _ffi_g * self.W_a2msiInh_AMPA
        W_a2msiInh_NMDA = _ffi_g * self.W_a2msiInh_NMDA
        W_v2msiInh_AMPA = _ffi_g * self.W_v2msiInh_AMPA
        W_v2msiInh_NMDA = _ffi_g * self.W_v2msiInh_NMDA
        W_msiInh2Exc_GABA = self.W_msiInh2Exc_GABA
        W_msi2out = self.W_msi2out

        # ---- ring-buffer local aliases ----
        buf_a2msi = self.buffer_a2msi
        buf_v2msi = self.buffer_v2msi
        # task #192 Phase A: REMOVED buf_inA_inh / buf_inV_inh ring-buffer aliases.
        buf_a2msi_inh = self.buffer_a2msi_inh
        buf_v2msi_inh = self.buffer_v2msi_inh
        buf_msi_inh2exc = self.buffer_msi_inh2exc
        buf_msi2out = self.buffer_msi2out
        buf_msi_rec = self.buffer_msi_rec   # delay-fix: recurrent MSI->MSI delay line

        pos_a2msi = self._delay_positions["buffer_a2msi"]
        pos_v2msi = self._delay_positions["buffer_v2msi"]
        # task #192 Phase A: REMOVED pos_inA_inh / pos_inV_inh ring-buffer cursors.
        pos_a2msi_inh = self._delay_positions["buffer_a2msi_inh"]
        pos_v2msi_inh = self._delay_positions["buffer_v2msi_inh"]
        pos_msi_inh2exc = self._delay_positions["buffer_msi_inh2exc"]
        pos_msi2out = self._delay_positions["buffer_msi2out"]
        pos_msi_rec = self._delay_positions["buffer_msi_rec"]   # delay-fix

        # task #27: when dt_correct_nmda is True, derive substep delays from
        # physical-ms attributes so the physical delay duration is dt-invariant.
        # Otherwise use the legacy substep-stored integer attributes.
        # task #192 Phase A: REMOVED delay_inA_inh / delay_inV_inh.
        if self.dt_correct_nmda and hasattr(self, 'conduction_delay_a2msi_ms'):
            delay_a2msi       = self._delay_substeps_from_ms(self.conduction_delay_a2msi_ms)
            delay_v2msi       = self._delay_substeps_from_ms(self.conduction_delay_v2msi_ms)
            delay_a2msi_inh   = self._delay_substeps_from_ms(self.conduction_delay_a2msi_inh_ms)
            delay_v2msi_inh   = self._delay_substeps_from_ms(self.conduction_delay_v2msi_inh_ms)
            delay_msi_inh2exc = self._delay_substeps_from_ms(self.conduction_delay_msi_inh2exc_ms)
            delay_msi2out     = self._delay_substeps_from_ms(self.conduction_delay_msi2out_ms)
            delay_msi_rec     = self._delay_substeps_from_ms(self.conduction_delay_msi_rec_ms)
        else:
            delay_a2msi = self.conduction_delay_a2msi
            delay_v2msi = self.conduction_delay_v2msi
            delay_a2msi_inh = self.conduction_delay_a2msi_inh
            delay_v2msi_inh = self.conduction_delay_v2msi_inh
            delay_msi_inh2exc = self.conduction_delay_msi_inh2exc
            delay_msi2out = self.conduction_delay_msi2out
            delay_msi_rec = self.conduction_delay_msi_rec

        zero_exc = torch.zeros((batch_size, self.n), device=self.device)
        zero_inh = torch.zeros((batch_size, self.n_inh), device=self.device)

        # ---- accumulate debug counters on GPU, sync once at end ----
        dbg_spk_A = torch.tensor(0.0, device=self.device)
        dbg_spk_V = torch.tensor(0.0, device=self.device)
        dbg_spk_Mi = torch.tensor(0.0, device=self.device)  # task#21/S1: PV/FS
        dbg_spk_M = torch.tensor(0.0, device=self.device)

        for sub_i in range(self.n_substeps):
            # --- Decay old currents ---
            # task #192 Phase B: I_M_gaba decays at tau_gaba (slow);
            # I_M (AMPA + NMDA-driven) decays at tau_ampa (fast).
            self.I_A.mul_(decay_factor)
            self.I_V.mul_(decay_factor)
            self.I_M.mul_(decay_factor)
            self.I_M_gaba.mul_(gaba_decay)
            if self.gaba_shunt_surr: self.I_M_gaba_surr_sh.mul_(gaba_decay)
            self.I_M_inh.mul_(decay_factor)
            self.I_O.mul_(decay_factor)

            # Add external input (split across substeps)
            self.I_A.add_(I_A_input * input_step_scale)
            self.I_V.add_(I_V_input * input_step_scale)

            # --- Ring-buffer reads (replaces deque popleft) ---
            # task #192 Phase A: REMOVED delayed_spikes_inA_inh / _inV_inh
            # reads (direct FF-inh shortcut rip'd).
            # task #327 — per-element afferent-arrival jitter on the READ. OFF (_aff_jitter_a is None) =>
            # scalar buf[pos] (byte-identical). ON => read each trial b at (pos - delay_base - jitter_b) % L
            # where L = ring length (= delay_base + margin); +jitter_b = older row = longer conduction delay.
            if delay_a2msi > 0:
                if self._aff_jitter_a is None:
                    delayed_spikes_a2msi = buf_a2msi[pos_a2msi]
                else:
                    _idx_a = (pos_a2msi - delay_a2msi - self._aff_jitter_a) % buf_a2msi.shape[0]
                    delayed_spikes_a2msi = buf_a2msi[_idx_a, self._aff_batch_arange]
            else:
                delayed_spikes_a2msi = zero_exc
            if delay_v2msi > 0:
                if self._aff_jitter_v is None:
                    delayed_spikes_v2msi = buf_v2msi[pos_v2msi]
                else:
                    _idx_v = (pos_v2msi - delay_v2msi - self._aff_jitter_v) % buf_v2msi.shape[0]
                    delayed_spikes_v2msi = buf_v2msi[_idx_v, self._aff_batch_arange]
            else:
                delayed_spikes_v2msi = zero_exc
            delayed_spikes_a2msi_inh = buf_a2msi_inh[pos_a2msi_inh] if delay_a2msi_inh > 0 else zero_exc
            delayed_spikes_v2msi_inh = buf_v2msi_inh[pos_v2msi_inh] if delay_v2msi_inh > 0 else zero_exc
            delayed_spikes_msi_inh2exc = buf_msi_inh2exc[pos_msi_inh2exc] if delay_msi_inh2exc > 0 else zero_inh
            delayed_spikes_msi2out = buf_msi2out[pos_msi2out] if delay_msi2out > 0 else zero_exc
            # delay-fix: recurrent MSI->MSI presynaptic spikes, delayed by
            # conduction_delay_msi_rec substeps (100 = 10 ms = 1 external step).
            delayed_spikes_msi_rec = buf_msi_rec[pos_msi_rec] if delay_msi_rec > 0 else zero_exc

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
            scale_F = self.nmda_std_scale  # task#104: env NMDA_STD_SCALE (default 0.8 byte-id).  # 0 = no STD, 1 = same as AMPA
            gate_A_nmda = 1.0 - scale_F * (1.0 - self.R_a)  # (B,n)
            gate_V_nmda = 1.0 - scale_F * (1.0 - self.R_v)

            pre_A_nmda = gate_A_nmda * delayed_spikes_a2msi  # (B,n)
            pre_V_nmda = gate_V_nmda * delayed_spikes_v2msi
            # ---------------------------------------------------------------

            nmda_a = F.linear(pre_A_nmda, W_a2msi_NMDA)
            nmda_v = F.linear(pre_V_nmda, W_v2msi_NMDA * self.v2msi_nmda_scale)   # latency #148: V-specific NMDA fwd-scale (default 1.0 == byte-identical)
            inc_m_exc = self.nmda_alpha * (nmda_a + nmda_v)  # unchanged
            # latency #153: V-specific FF-NMDA time-course split-enabler. Default
            # (tau_nmda_v==tau_nmda) runs the ORIGINAL shared-pool path verbatim
            # => bit-identical. Split path gives the V->MSI FF-NMDA its own decay
            # (nmda_m_v) while A keeps tau_nmda; global gNMDA, (mg_A+mg_V) Mg-block,
            # A-pathway, recurrent + inhibitory NMDA all UNTOUCHED.
            if self.tau_nmda_v == self.tau_nmda:
                self.nmda_m.mul_(nmda_decay)
                self.nmda_m.add_(inc_m_exc)
                nmda_m_eff = self.nmda_m
            else:
                self.nmda_m.mul_(nmda_decay)
                self.nmda_m.add_(self.nmda_alpha * nmda_a)
                self.nmda_m_v.mul_(nmda_decay_v)
                self.nmda_m_v.add_(self.nmda_alpha * nmda_v)
                nmda_m_eff = self.nmda_m + self.nmda_m_v

            # Dend coupling
            d_va = self.dend_coupling_alpha * (self.v_msi - self.v_dend_A) / self.tau_m
            d_vv = self.dend_coupling_alpha * (self.v_msi - self.v_dend_V) / self.tau_m
            self.v_dend_A += self.dt * d_va
            self.v_dend_V += self.dt * d_vv

            dv_nmda = ((self.v_msi + self.nmda_vrest_offset) - self.v_nmda) / self.tau_nmdaVolt
            self.v_nmda += self.dt * dv_nmda
            mg_A = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_A - self.mg_vhalf)))
            mg_V = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_V - self.mg_vhalf)))
            I_nmda = self.gNMDA * nmda_m_eff * (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)   # latency #153: nmda_m_eff == self.nmda_m at default (bit-identical); == nmda_m(A) + nmda_m_v(V) in split path
            I_nmda_step = self.gNMDA * inc_m_exc * (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)
            I_nmda_lp = self.gNMDA * self.nmda_m * (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)

            # task #135: unified linear dt-scaling (matches AMPA's existing fix).
            # `dt_linear_scale = dt / 0.1` precomputed at L1997. At canonical
            # dt=0.1 this is 1.0 → byte-identical to bare `add_(I_nmda)` → legacy
            # ckpts produce paper biology unchanged. At other dt, per-substep
            # injection scales linearly so total per-ms injection is preserved.
            # Replaces task #121's exp-Euler Form 2 (`1 - exp(-dt/tau_syn)`),
            # which was ≈0.0392 at dt=0.1 and broke paper output on legacy ckpts.
            self.I_M.add_(I_nmda * dt_linear_scale)

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

            # task #192 Phase A: REMOVED the direct A/V -> MSI_exc inhibitory
            # injection block (`I_inA_inh = F.linear(..., W_inA_inh); I_inV_inh
            # = F.linear(..., W_inV_inh); I_M.sub_(g_FFinh * (I_inA_inh +
            # I_inV_inh))`). Biologically unjustified shortcut per researcher
            # #191 §1. Disynaptic inhibition via A/V -> MSI_inh -> MSI_exc
            # remains (W_a2msiInh_AMPA/NMDA + W_msiInh2Exc_GABA below).

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
            self.nmda_m_inh.mul_(nmda_decay_inh)
            # task #14 (debugger-validated, patch_trace.py block D): presynaptic
            # short-term depression on the afferent->interneuron NMDA branch,
            # mirroring the exc-side NMDA STD gate (L2384: gate = 1 - scale_F*(1-R)).
            # Reuses the SHARED vesicle resource R_*_inh already depleting the
            # co-located AMPA-in branch (u_*_inh=0.2); the prior NMDA omission was a
            # structural AMPA/NMDA co-release asymmetry, not biology. scale_F=0.8.
            _sF_inh = 0.8
            _gate_a_nmda_inh = 1.0 - _sF_inh * (1.0 - self.R_a_inh)
            _gate_v_nmda_inh = 1.0 - _sF_inh * (1.0 - self.R_v_inh)
            self.nmda_m_inh.add_(self.nmda_alpha * (_gate_a_nmda_inh * raw_inp_a_NMDA + _gate_v_nmda_inh * raw_inp_v_NMDA))

            d_viA = self.dend_coupling_alpha * (self.v_msi_inh - self.v_dend_inhA) / self.tau_m
            d_viV = self.dend_coupling_alpha * (self.v_msi_inh - self.v_dend_inhV) / self.tau_m
            self.v_dend_inhA += self.dt * d_viA
            self.v_dend_inhV += self.dt * d_viV

            dv_nmda_inh = ((self.v_msi_inh + self.nmda_vrest_offset) - self.v_nmda_inh) / self.tau_nmdaVolt
            self.v_nmda_inh += self.dt * dv_nmda_inh
            mg_iA = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_inhA - self.mg_vhalf_inh)))
            mg_iV = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_inhV - self.mg_vhalf_inh)))
            # task#21/E2/S2: INH-plateau accumulator — battery single-volley probe ONLY
            # (_panel_inh_accum is None during training => strict no-op, trajectory-neutral)
            if self._panel_inh_accum is not None:
                _pa = self._panel_inh_accum
                _pa['mg_iA'].append(mg_iA.mean().item())
                _pa['mg_iV'].append(mg_iV.mean().item())
                _pa['mg_iA_on'].append((mg_iA > 0.5).float().mean().item())
                _pa['mg_iV_on'].append((mg_iV > 0.5).float().mean().item())
                _pa['v_dend_inhA'].append(self.v_dend_inhA.mean().item())
                _pa['v_dend_inhV'].append(self.v_dend_inhV.mean().item())
            I_nmda_inh = self.gNMDA * self.nmda_m_inh * (mg_iA + mg_iV) * (self.Erev_nmda - self.v_msi_inh)
            # task #135: unified linear dt-scaling for inhibitory NMDA (same as
            # the I_M site above; `dt_linear_scale = dt / 0.1` precomputed at
            # L1997). Identity at dt=0.1, preserves per-ms injection elsewhere.
            self.I_M_inh.add_(I_nmda_inh * dt_linear_scale)

            # MSI_inh->MSI_ex
            I_M_inh2exc = F.linear(delayed_spikes_msi_inh2exc, W_msiInh2Exc_GABA)
            # task #125: REVERTED task #123 FIX 3 (`* dt_linear_scale`). Same
            # rationale as FIX 2 revert above — debugger #124 found GABA recurrent
            # source-scaling also breaks the natural homeostatic balance.
            # task #192 Phase B: route disynaptic GABA into I_M_gaba (tau_gaba
            # decay = 50ms biology) instead of fast I_M (tau_ampa = 2.5ms).
            # Magnitude is added as positive; subtracted from I_M at integration.
            self.I_M_gaba.add_(self.pv_gaba_scale * I_M_inh2exc)   # PV_GABA_SCALE lever (default 1.0 => x1.0 bit-exact; ASD ckpt restores <1 to measure under reduced PV feed-forward inhibition)

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
            # task #192 Phase B: net current is (I_M - I_M_gaba) where I_M
            # holds AMPA/NMDA-driven fast excitation (decays at tau_ampa=2.5ms)
            # and I_M_gaba holds tonic + disynaptic + surround GABAergic
            # inhibition magnitude (decays at tau_gaba=50ms slow IPSC biology).
            I_surr_shunt = self.k_shunt_surr * self.I_M_gaba_surr_sh * (self.E_gaba - self.v_msi)   # <=0 divisive surround; identically 0 when OFF
            self._last_I_surr_shunt = I_surr_shunt.detach()   # task #55: expose exact pre-spike-reset surround current for shunt-aware E/I (val36 ei_probe_route_c); ==0 when OFF
            dVM = (0.04 * self.v_msi.pow(2) + 5.0 * self.v_msi + 140.0
                   - self.u_msi + (self.I_M - self.I_M_gaba + I_surr_shunt))
            # task#130 Form-A: dVm/dt-ADAPTIVE spike threshold (Azouz & Gray 2000) — the missing cellular
            # substrate for multisensory ONSET-latency facilitation (#106/#125). dVM IS the instantaneous
            # dv_msi/dt (mV/ms), read here PRE-Euler/PRE-reset so the -65mV reset never pollutes the trace.
            # A coincident, steeper-rising A+V input lowers the effective threshold -> earlier first spike.
            # k_dvdt=0 (default) => _formA_on False => v_thresh_M == spike_threshold => byte-identical static rule.
            if self._formA_on:
                dvdt_pos = torch.clamp(torch.relu(dVM), 0.0, self.dvdt_cap)                          # rectify (only depolarization lowers theta) + cap the regenerative upstroke
                self.dvdt_trace_msi += (self.dt / self.tau_dvdt) * (dvdt_pos - self.dvdt_trace_msi)  # low-pass relu(dv/dt), tau_dvdt ms
                v_thresh_M = torch.clamp(spike_threshold - self.k_dvdt * self.dvdt_trace_msi,
                                         self.v_thresh_floor, spike_threshold)
            else:
                v_thresh_M = spike_threshold
            self.v_msi += self.dt * dVM
            self.u_msi += self.dt * (self.aM * (self.bM * self.v_msi - self.u_msi))
            spike_mask_M = (self.v_msi >= v_thresh_M)
            new_sM = spike_mask_M.float()
            if self._fs_probe_on:   # latency #152: substep-res max-v_msi PRE-reset (sub-threshold FOOT probe; appends a detached scalar peak, mutates NO dynamics => byte-identical when off)
                self._v_msi_substep_trace.append(self.v_msi.detach().amax())
            self.v_msi.masked_fill_(spike_mask_M, self.cM)
            self.u_msi[spike_mask_M] += self.dM
            # task#114 STEP-1 substep-resolution first-spike probe (pure instrumentation: reads spike_mask_M,
            # mutates NO dynamics state; gated OFF by default so every other path stays byte-identical). Records
            # per trial the GLOBAL substep index (x dt = 0.1ms) of the first MSI-population spike, so that
            # response_latency_routec.py resolves the sub-frame dV/dt latency advance the 10ms frame ruler hides.
            if self._fs_probe_on:
                _fs_newly = (self._first_spike_substep < 0) & spike_mask_M.any(dim=1)
                self._first_spike_substep = torch.where(
                    _fs_newly,
                    torch.full_like(self._first_spike_substep, self._fs_substep_clock),
                    self._first_spike_substep)
                if self._formA_on:   # task#130 anchor: capture the dV/dt low-pass trace (rate-of-rise) at the crossing
                    self._fs_dvdt_trace = torch.where(   # substep, per trial = max over neurons (the first-crossing unit)
                        _fs_newly, self.dvdt_trace_msi.amax(dim=1), self._fs_dvdt_trace)
                self._fs_substep_clock += 1

            # (A) compute surround inhibition current
            I_latM = torch.mm(new_sM, self.W_MSI_inh)  # shape (B, n)
            # (B) apply it
            # task #192 Phase B: route Mexican-hat lateral GABA into I_M_gaba
            # (tau_gaba decay = 50ms biology) instead of fast I_M. Magnitude
            # is added as positive; subtracted from I_M at integration.
            if self.gaba_shunt_surr:
                self.I_M_gaba_surr_sh.add_(self.g_GABA * I_latM)   # surround -> shunting accumulator (divisive form)
            else:
                self.I_M_gaba.add_(self.g_GABA * I_latM)           # subtractive (byte-identical default)

            # ── recurrent MSI->MSI excitation — Increment 1 (INERT at g_rec=0) ──
            # Excitatory sibling of the surround inhibition just above: this
            # substep's MSI spikes (new_sM) drive a narrow local Gaussian
            # (W_MSI_exc) carrying co-localised AMPA + NMDA with driving force
            # and a single FF dendritic Mg-block gate (mg_A) — mirroring the FF
            # A/V->MSI NMDA physics, only the conductance SOURCE differs (MSI
            # spikes, not afferents) — injected into the fast excitatory channel
            # I_M. (The recurrent input shares the FF dendritic Mg gate; a
            # dedicated recurrent dendritic compartment is an activation-increment
            # option, inert here at g_rec=0.) The whole
            # block is gated on g_rec: at the default 0.0 it does not execute, so
            # I_M / nmda_m_rec are untouched and inference is bit-identical to the
            # pre-recurrence build. dt_linear_scale keeps the injection
            # dt-invariant exactly like the FF AMPA/NMDA sites. No plasticity yet.
            if self.g_rec != 0.0:
                # delay-fix EDIT #1: inject from the DELAYED recurrent spikes
                # (presynaptic MSI activity from 1 external step / 10 ms ago) instead
                # of the instantaneous new_sM, so pre leads post on the recurrent path.
                g_rec_syn = F.linear(delayed_spikes_msi_rec, self.W_MSI_exc)   # (B,n) conductance
                I_rec_ampa = (self.gAMPA * (self.rec_ampa_frac * g_rec_syn)
                              * (self.Erev_ampa - self.v_msi))
                self.nmda_m_rec.mul_(nmda_decay)
                self.nmda_m_rec.add_(self.nmda_alpha * (self.rec_nmda_frac * g_rec_syn))
                # single post-synaptic Mg-unblock gate. v_dend_A ≡ v_dend_V are
                # degenerate (identical init + identical passive coupling to the same
                # soma, no differential A/V dendritic drive) ⇒ mg_A ≡ mg_V, so the FF
                # site's (mg_A+mg_V) would be a literal 2× double-count on this NEW
                # untuned recurrent path. Use ONE gate (mg_A) — each FF NMDA pathway
                # is itself gated by exactly one Mg factor — preserving the intended
                # 0.75:0.25 recurrent NMDA:AMPA balance.
                I_rec_nmda = (self.gNMDA * self.nmda_m_rec * mg_A
                              * (self.Erev_nmda - self.v_msi))
                self.I_M.add_(self.g_rec * (I_rec_ampa + I_rec_nmda) * dt_linear_scale)

            # ── E/I component recording (separate synaptic currents) ──
            if self._ei_record is not None:
                # Excitatory onto MSI excitatory
                _I_E_ampa = torch.clamp(I_AMPA_curr, min=0.0)
                # task #135: mirror the unified linear dt-scaling applied at
                # the NMDA->I_M injection site so the probe records the same
                # scaled current that is actually injected (E/I symmetry).
                _I_E_nmda = torch.clamp(I_nmda * dt_linear_scale, min=0.0)
                _I_E = _I_E_ampa + _I_E_nmda
                # Inhibitory onto MSI excitatory
                # task #192 Phase A: _I_I_ff is now ZERO by construction (the
                # direct A/V -> MSI_exc shortcut was rip'd). Key is retained in
                # the record dict for backward compat with downstream consumers
                # (EI_balance_test.py / run_ei_balance.py reference "FFInh"),
                # but the contributed current is genuinely zero now.
                _I_I_ff = torch.zeros_like(_I_E)
                _I_I_recur = torch.clamp(I_M_inh2exc, min=0.0)
                if self.gaba_shunt_surr:
                    _I_I_lat = torch.clamp(-I_surr_shunt, min=0.0)        # record the integrated shunting surround current
                else:
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
                # task #192 Phase B: I_total is the NET current onto MSI exc,
                # i.e. fast (AMPA + NMDA + tonic stays-in-I_M dropped) minus
                # slow GABA state. Mirrors what Izhikevich actually integrates.
                if self.gaba_shunt_surr:
                    I_total = (self.I_M - self.I_M_gaba + I_surr_shunt).detach()
                else:
                    I_total = (self.I_M - self.I_M_gaba).detach()
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
                # task #192 Phase A: inh_FF dropped (direct FF-inh shortcut rip'd).
                inh_lat = (self.g_GABA * I_latM).sum().item()
                inh_recur = I_M_inh2exc.sum().item()


            if return_spike_sum:  # ****
                sum_sM += new_sM  # ****

            # task #188: REMOVED iSTDP post_i_trace decay+update (no iSTDP rule).

            self.post_rate_avg.mul_(1.0 - rate_alpha).add_(rate_alpha * new_sM)

            # MSI inh
            dVMi = (0.04 * self.v_msi_inh.pow(2) + 5.0 * self.v_msi_inh + 140.0 - self.u_msi_inh + self.I_M_inh)
            self.v_msi_inh += self.dt * dVMi
            self.u_msi_inh += self.dt * (self.aMi * (self.bMi * self.v_msi_inh - self.u_msi_inh))
            spike_mask_Mi = (self.v_msi_inh >= spike_threshold) & (self.msi_inh_refrac <= 0)
            new_sMi = spike_mask_Mi.float()
            self.v_msi_inh.masked_fill_(spike_mask_Mi, self.cMi)
            self.u_msi_inh[spike_mask_Mi] += self.dMi
            self.msi_inh_refrac.masked_fill_(spike_mask_Mi, float(self.msi_inh_refrac_substeps))
            self.v_msi_inh.masked_fill_(self.msi_inh_refrac > 0, self.cMi)
            self.msi_inh_refrac.add_(-1.0).clamp_(min=0.0)

            # Out
            dVO = (0.04 * self.v_out.pow(2) + 5.0 * self.v_out + 140.0 - self.u_out + self.I_O)
            self.v_out += self.dt * dVO
            self.u_out += self.dt * (self.aO * (self.bO * self.v_out - self.u_out))
            spike_mask_O = (self.v_out >= spike_threshold)
            new_sO = spike_mask_O.float()
            self.v_out.masked_fill_(spike_mask_O, self.cO)
            self.u_out[spike_mask_O] += self.dO

            # task #188 (simple-pathway §4.1): REMOVED iSTDP Vogels-Abbott block.
            # Inhibitory weights W_inA_inh / W_inV_inh are now FIXED at init (2.0)
            # per spec §3.1. No plasticity on inh side — biology-defensible at FSTS
            # scope per Sooksawate 2011, Mize 1988, Carrasco-Razak 2011 timescale.

            if valid_mask is not None:
                mask_sub = valid_mask.view(-1, 1)
                new_sA *= mask_sub
                new_sV *= mask_sub
                new_sM *= mask_sub
                new_sMi *= mask_sub
                new_sO *= mask_sub

            # task#385 SOM/Martinotti facilitation-in-time on the interneuron->exc GABA edge (debugger #384 §2C.3).
            # F_gaba decays toward baseline U at tau_facil and JUMPs toward 1 on each interneuron spike (TM
            # facilitation); rel_sMi = F_gaba * new_sMi is the facilitated release fraction written to the delay
            # buffer (near-zero early, grows late). OFF => rel_sMi = new_sMi (same tensor) => byte-identical.
            # new_sMi stays RAW downstream (rate counter + iSTDP pre-trace, §2D) so the homeostat rebalances the
            # SCALAR W_msiInh2Exc_GABA without cancelling the temporal facilitation shape.
            if self.facil_gaba_on:
                self.F_gaba += (self.U_facil_gaba - self.F_gaba) * (self.dt / self.tau_facil_gaba)
                self.F_gaba += self.U_facil_gaba * (1.0 - self.F_gaba) * new_sMi
                rel_sMi = self.F_gaba * new_sMi
            else:
                rel_sMi = new_sMi          # OFF => byte-identical

            # Ring-buffer writes (replaces deque append)
            # task #192 Phase A: REMOVED inA_inh / inV_inh writes.
            if delay_a2msi > 0:
                buf_a2msi[pos_a2msi].copy_(new_sA)   # write current spike (all trials) at pos
                pos_a2msi = (pos_a2msi + 1) % buf_a2msi.shape[0]   # task #327: advance mod ring length (==delay when OFF)
            if delay_v2msi > 0:
                buf_v2msi[pos_v2msi].copy_(new_sV)
                pos_v2msi = (pos_v2msi + 1) % buf_v2msi.shape[0]
            if delay_a2msi_inh > 0:
                buf_a2msi_inh[pos_a2msi_inh].copy_(new_sA)
                pos_a2msi_inh = (pos_a2msi_inh + 1) % delay_a2msi_inh
            if delay_v2msi_inh > 0:
                buf_v2msi_inh[pos_v2msi_inh].copy_(new_sV)
                pos_v2msi_inh = (pos_v2msi_inh + 1) % delay_v2msi_inh
            if delay_msi_inh2exc > 0:
                buf_msi_inh2exc[pos_msi_inh2exc].copy_(rel_sMi)  # task#385: facilitated release (OFF: rel_sMi==new_sMi)
                pos_msi_inh2exc = (pos_msi_inh2exc + 1) % delay_msi_inh2exc
            if delay_msi2out > 0:
                buf_msi2out[pos_msi2out].copy_(new_sM)
                pos_msi2out = (pos_msi2out + 1) % delay_msi2out
            # delay-fix: push this substep's MSI spikes into the recurrent delay line.
            if delay_msi_rec > 0:
                buf_msi_rec[pos_msi_rec].copy_(new_sM)
                pos_msi_rec = (pos_msi_rec + 1) % delay_msi_rec

            sA, sV, sM, sMi, sO = new_sA, new_sV, new_sM, new_sMi, new_sO

            self._latest_sA = sA
            self._latest_sV = sV
            self._latest_sMSI = sM
            self._latest_sMSI_inh = sMi

            # ---------- task #192 Phase E: D'Amour-Froemke iSTDP on the
            # MSI_inh -> MSI_exc GABA edge (W_msiInh2Exc_GABA). Gated by
            # epoch_idx > 25 to match the MSI plasticity staging used by
            # apply_topographic_anchor_msi / _msi_inh; the iSTDP rule
            # balances inh against an exc target that itself only learns
            # post-stage-2. Trace shapes:
            #   pre_trace_msiInh2Exc  : (B, n_inh)  -- presyn = MSI_inh
            #   post_trace_msiInh2Exc : (B, n)      -- postsyn = MSI_exc
            # W_msiInh2Exc_GABA shape : (n, n_inh)
            if getattr(self, 'plasticity_enabled', True) and epoch_idx > 25:
                pre_decay  = 1.0 - self.dt / self.tau_istdp_pre
                post_decay = 1.0 - self.dt / self.tau_istdp_post
                self.pre_trace_msiInh2Exc.mul_(pre_decay)
                self.pre_trace_msiInh2Exc.add_(self._latest_sMSI_inh)
                self.post_trace_msiInh2Exc.mul_(post_decay)
                self.post_trace_msiInh2Exc.add_(self._latest_sMSI)

                # symmetric Hebbian iSTDP weight update:
                #   dW = eta * ( (post_trace - base) (x) pre_spike
                #              + (pre_trace  - base) (x) post_spike )
                # broadcast across (B, n, n_inh) then mean over batch.
                post_term = (self.post_trace_msiInh2Exc - self.istdp_baseline)
                pre_term  = (self.pre_trace_msiInh2Exc  - self.istdp_baseline)
                dW_istdp = self.eta_istdp * (
                    post_term.unsqueeze(-1) * self._latest_sMSI_inh.unsqueeze(1)
                    + pre_term.unsqueeze(1)  * self._latest_sMSI.unsqueeze(-1)
                ).mean(0)
                self.W_msiInh2Exc_GABA.data.add_(dW_istdp)
                self.W_msiInh2Exc_GABA.data.clamp_(min=0.0, max=self.W_gaba_clamp)

            # ---------- epoch-level debug counters (GPU accumulate) ----
            dbg_spk_A += sA.sum()
            dbg_spk_V += sV.sum()
            dbg_spk_Mi += sMi.sum()  # task#21/S1: PV/FS spikes (mirror of sM, post-mask)
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
                    # task #192 Phase D: FROZEN per sci_inhibition_phase_d_revision.md
                    # §1.3 (researcher post-debugger #193 forensic). The Oja form had
                    # fixed point W = G_max = 1.0 (Gaussian peak), divorced from any
                    # biology operating point; biology B1-B5 rule out plasticity at
                    # FF->PV+ (Lamsa 2007 anti-Hebbian; Le Roux 2013 NMDA FB-only;
                    # Turrigiano 2012 no scaling at exc->GABAergic; Carrasco 2011
                    # PNN-locked adult SC). W_a2msiInh / W_v2msiInh now frozen at
                    # Whyland-Bickford 2018 init throughout training. Function
                    # apply_topographic_anchor_msi_inh remains as dead code (L377-412)
                    # for easy revert if biology re-opens.

            self.step_counter += 1

        # ------ write back ring-buffer positions ------
        # task #192 Phase A: REMOVED inA_inh / inV_inh position write-back.
        self._delay_positions["buffer_a2msi"] = pos_a2msi
        self._delay_positions["buffer_v2msi"] = pos_v2msi
        self._delay_positions["buffer_a2msi_inh"] = pos_a2msi_inh
        self._delay_positions["buffer_v2msi_inh"] = pos_v2msi_inh
        self._delay_positions["buffer_msi_inh2exc"] = pos_msi_inh2exc
        self._delay_positions["buffer_msi2out"] = pos_msi2out
        self._delay_positions["buffer_msi_rec"] = pos_msi_rec   # delay-fix

        # ------ sync debug counters (one GPU->CPU transfer) ------
        self._dbg_spk_A += dbg_spk_A.item()
        self._dbg_spk_V += dbg_spk_V.item()
        self._dbg_spk_MSI += dbg_spk_M.item()
        self._dbg_spk_Mi += dbg_spk_Mi.item()  # task#21/S1: PV/FS host-sync
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

            # task #185 (minimal-arch §5.1): REMOVED slow_synaptic_scaling call block.
            # Function definition retained at L529-538 per Q-M3 for cheap revert.
            # Rationale: per #179 §3.4 the broken 60,000× cadence made the scaling
            # compete with iSTDP at the wrong timescale. Under the minimal architecture
            # parsimony principle, the function is OMITTED entirely (FSTS training
            # window doesn't reach Turrigiano biology timescale anyway).

        # task #185 (minimal-arch §5.1): REMOVED AGC fast + slow loops.
        # Rationale: AGC is an engineering hack (single-scalar PI controller on
        # network-mean current); no biological equivalent. iSTDP is the
        # biology-conformant E/I-balance regulator at the seconds-minutes
        # timescale where this work matters.

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

        W = getattr(self, W_attr)
        _sb = getattr(self, '_softbound_wmax', None)
        if _sb is not None and W_attr in _sb:
            _mu = getattr(self, '_softbound_mu', 1.0)
            _Wmax = _sb[W_attr]
            ltp = torch.zeros_like(W)
            ltd = torch.zeros_like(W)
            for b in range(B):
                ltp += torch.ger(post_spk[b], pre_trace[b])
                ltd += torch.ger(post_trace[b], pre_spk[b])
            _g_ltp = (_Wmax - W).clamp(min=0.0).pow(_mu)
            _g_ltd = W.clamp(min=0.0).pow(_mu)
            dW = A_plus * _g_ltp * ltp - A_minus * _g_ltd * ltd
        else:
            dW = torch.zeros_like(W)
            for b in range(B):
                dW += A_plus * torch.ger(post_spk[b], pre_trace[b]) \
                      - A_minus * torch.ger(post_trace[b], pre_spk[b])
        dW.mul_(lr / B)

        # Apply via the positive reparam
        self._p_add(W_attr, dW)

        # task#21/E3/S3: Tap-A FF dW-flux capture (read-only; _panel_dW_accum is
        # None unless run_training set it this epoch => strict no-op otherwise).
        # Centralised here so all 6 FF STDP weights are caught in one place.
        if self._panel_dW_accum is not None and W_attr in self._panel_dW_accum:
            self._panel_dW_accum[W_attr] += dW.abs().sum().item()

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
                temporal_jitter_max=2,
                # task #100 — INTRINSIC common-mode A-V latency jitter in TRAINING (same knob as the
                # measurement path). self.sigma_dL_frames == 0.0 -> byte-identical to pristine.
                sigma_dL_frames=float(self.sigma_dL_frames),
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

                # Increment 3: recurrent MSI->MSI excitatory STDP, gated post-ep25
                # to match the MSI plasticity staging (the g_rec conductance turns
                # on at the same epoch gate). Faithful mirror of the FF->MSI
                # excitatory STDP above: SAME stdp_update_batch rule, lr, and trace
                # machinery — the ONLY change is the recurrent source/target
                # (pre == post == MSI population spikes sMSI). The learned iSTDP on
                # W_msiInh2Exc_GABA is the stabilizer; W_MSI_exc is intentionally
                # NOT registered in _softbound_wmax (no added homeostatic cap), so
                # it uses the additive STDP branch under the universal _p_add
                # guardrail. Diagonal re-zeroed each step to preserve the
                # no-autapse structure (init fill_diagonal_(0)).
                if epoch_idx > 25:
                    # delay-fix EDIT #2: the recurrent STDP pre is the PREVIOUS external
                    # step's MSI spikes (delayed by 1 external step), so pre leads post.
                    # This breaks the post==pre==sMSI antisymmetric LTP=LTD cancellation
                    # (debugger H1; matches the §B harness lag=1 proof). post stays the
                    # current-step sMSI; the pre-trace is built off the delayed pre inside
                    # stdp_update_batch. _prev is None (=> zeros) at each sequence start.
                    pre_spk_rec = (self._prev_sMSI_rec if self._prev_sMSI_rec is not None
                                   else torch.zeros_like(sMSI))
                    self.pre_trace_msi_rec, self.post_trace_msi_rec, _ = self.stdp_update_batch(
                        'W_MSI_exc',
                        post_spk=sMSI,
                        pre_spk=pre_spk_rec,
                        post_trace=self.post_trace_msi_rec,
                        pre_trace=self.pre_trace_msi_rec,
                        lr=self.lr_msi,
                        debug=debug)
                    self.W_MSI_exc.data.fill_diagonal_(0.0)
                    self._prev_sMSI_rec = sMSI.detach()

                # --- OPTIONAL: re-normalise AMPA/NMDA split -------------------
                # Task #66 fix: migrated from set_param_weight()/parametrizations
                # to direct .copy_() ops with _p_add-style clamp bounds
                # (eps=1e-9, abs_cap=5.0). The parametrize wrappers were removed
                # in commit 1644af6 (de-parametrize refactor); this site was
                # missed during that migration. See debug_dt/DIAGNOSTIC_REPORT_task65.md.
                with torch.no_grad():
                    # A -> MSI connections — redistribute 25/75 AMPA/NMDA
                    if getattr(self, '_softbound_wmax', None):
                        _cap_aA = self._softbound_wmax['W_a2msi_AMPA']; _cap_aN = self._softbound_wmax['W_a2msi_NMDA']
                        _cap_vA = self._softbound_wmax['W_v2msi_AMPA']; _cap_vN = self._softbound_wmax['W_v2msi_NMDA']
                    else:
                        _cap_aA = _cap_aN = _cap_vA = _cap_vN = 5.0
                    W_tot_a = self.W_a2msi_AMPA + self.W_a2msi_NMDA
                    self.W_a2msi_AMPA.copy_((0.25 * W_tot_a).clamp_(min=1e-9, max=_cap_aA))
                    self.W_a2msi_NMDA.copy_((0.75 * W_tot_a).clamp_(min=1e-9, max=_cap_aN))

                    # V -> MSI connections — redistribute 25/75 AMPA/NMDA
                    W_tot_v = self.W_v2msi_AMPA + self.W_v2msi_NMDA
                    self.W_v2msi_AMPA.copy_((0.25 * W_tot_v).clamp_(min=1e-9, max=_cap_vA))
                    self.W_v2msi_NMDA.copy_((0.75 * W_tot_v).clamp_(min=1e-9, max=_cap_vN))

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
        If *enable* is True, this zeroes out the MSI_inh -> MSI_ex GABA gate
        and freezes inhibitory plasticity, leaving the Mexican-hat surround
        (W_MSI_inh + g_GABA) untouched. Call again with False to restore.

        task #192 Phase A: removed the W_inA_inh / W_inV_inh zero calls
        (direct FF-inh shortcut no longer exists).
        """
        z = 0.0 if enable else 1.0
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
        g_FFinh=net.g_FFinh,  # task#192 Phase A: orphan attribute (no longer functional)
        msiInh_input_gain=net.msiInh_input_gain,  # task#102 (B): persistent FFI-recruitment forward-multiplier (washout-proof; restored by load_ckpt + agreement-asserted)

        # MSI surround-inhibition hyper-params
        R_near=net.R_near,
        # task#192 Phase A: eta_H / eta_AH stripped (Mexican-hat learning
        # rates were never used in any update loop; attributes removed).

        # synaptic-current / receptor time-constants
        gNMDA=net.gNMDA,
        tau_syn=net.tau_syn,
        # task #192 Phase B: explicit AMPA / GABA tau split.
        tau_ampa=net.tau_ampa,
        tau_gaba=net.tau_gaba,
        # task#385 SOM/Martinotti facilitation-in-time; ckpt = source of truth (val36 restores -> MEASURE honors training)
        facil_gaba_on=net.facil_gaba_on, tau_facil_gaba=net.tau_facil_gaba, U_facil_gaba=net.U_facil_gaba,
        tau_nmda=net.tau_nmda,
        nmda_alpha=net.nmda_alpha,
        mg_k=net.mg_k,
        conduction_delay_v2msi=net.conduction_delay_v2msi,  # LATENCY-FIX V-advance lever (substeps); saved so the CKPT is the source of truth -> MEASURE honors the trained V arrival delay (load_ckpt restores + recomputes _ms/_inh)
        Erev_nmda=net.Erev_nmda,
        tau_nmdaVolt=net.tau_nmdaVolt,
        v_nmda_rest=net.v_nmda_rest,
        nmda_vrest_offset=net.nmda_vrest_offset,
        mg_vhalf=net.mg_vhalf,
        dend_coupling_alpha=net.dend_coupling_alpha,

        # Tsodyks–Markram STP
        tau_rec=net.tau_rec,
        tau_fac=net.tau_fac,
        u_stp_a=net.u_stp_a,            # task#104: TM-STD utilization (A/V) + NMDA-STD depth, env-exposed decouple knobs
        u_stp_v=net.u_stp_v,
        nmda_std_scale=net.nmda_std_scale,
        gAMPA_LP=net.gAMPA_LP,          # task#116: fast-AMPA-LP gain (orthogonal fast-exc rate lever, #80/#81), env GAMPA_LP, default 1.0 byte-identical
        k_dvdt=net.k_dvdt,              # task#130: Form-A dVm/dt-adaptive threshold (Azouz&Gray 2000) slope, env K_DVDT, default 0.0 byte-identical
        tau_dvdt=net.tau_dvdt,          # task#130: Form-A dV/dt low-pass window (ms), env TAU_DVDT, default 3.0
        v_thresh_floor=net.v_thresh_floor,  # task#130: Form-A effective-threshold clamp floor (mV), env V_THRESH_FLOOR, default 20.0
        dvdt_cap=net.dvdt_cap,          # task#130: Form-A relu(dv/dt) cap (mV/ms), env DVDT_CAP, default 50.0

        # iSTDP homeostasis
        rho0=net.rho0,
        eta_i=net.eta_i,
        tau_post_i=net.tau_post_i,
        rate_avg_tau=net.rate_avg_tau,

        # task #192 Phase E: D'Amour-Froemke symmetric iSTDP on the
        # MSI_inh -> MSI_exc GABA edge (W_msiInh2Exc_GABA). See spec
        # sci_inhibition_functional_role.md §5.4b.
        tau_istdp_pre=net.tau_istdp_pre,
        tau_istdp_post=net.tau_istdp_post,
        eta_istdp=net.eta_istdp,
        istdp_baseline=net.istdp_baseline,
        W_gaba_clamp=net.W_gaba_clamp,
        pv_gaba_scale=net.pv_gaba_scale,   # ASD lever: disynaptic PV->exc GABA scale (symmetry with trainer; round-trippable)

        # conduction delays outside the constructor
        # task#192 Phase A: removed conduction_delay_inA_inh / _inV_inh
        # (paired with direct FF-inh shortcut rip; attributes deleted).
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


def generate_two_event_offset_seq(loc, T=60, D=5, offset=0, space_size=180,
                                  sigma_dL_frames=0.0, rng=None):
    """
    A positive offset → visual lags audio by <offset> macro steps (10 ms each);
    a negative offset → visual leads; 0 → simultaneous.

    task #99 — correlated trial-timing noise (common-mode A-V latency jitter):
    when sigma_dL_frames > 0, draw ONE shared latency jitter ΔL ~ N(0, sigma_dL_frames)
    per call (= per trial; common-mode across the whole A/V representation built from this one
    sequence) and add it to the relative A-V onset: effective_offset = round(offset + ΔL).
    Quantized to the 10-ms macro-step grid (the stimulus is built per frame), matching the
    debugger #98 H4 correlated proof (integer-frame kernel). `rng` (np.random.Generator) makes
    the draw reproducible-but-independent per trial; falls back to np.random when None.
    DEFAULT sigma_dL_frames=0.0 -> block skipped -> byte-identical to pristine.
    """
    if sigma_dL_frames and sigma_dL_frames > 0.0:
        _draw = rng.normal(0.0, float(sigma_dL_frames)) if rng is not None \
            else np.random.normal(0.0, float(sigma_dL_frames))
        offset = int(round(offset + _draw))
        _lim = T - D                                   # keep later onset + D within T (no IndexError)
        offset = max(-_lim, min(_lim, offset))
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


def _write_panel_outputs(rows, out_dir, seed):
    """task#21/E6: write per-epoch panel rows -> panel_seed{seed}.csv + .npz."""
    import csv, os
    if not rows:
        print(f"[panel] no rows to write for seed {seed}")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"panel_seed{seed}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    cols = {k: np.array([r.get(k, float('nan')) for r in rows], dtype=float) for k in keys}
    npz_path = os.path.join(out_dir, f"panel_seed{seed}.npz")
    np.savez(npz_path, **cols)
    print(f"[panel] wrote {csv_path} + {npz_path} ({len(rows)} rows, {len(keys)} cols)")


def run_training(
        batch_size=1000,
        n_unsup_epochs=80,
        seed=None,
        panel=False,
        panel_out_dir=".",
):
    """
    Main training run
    """
    # task#21/E7: deterministic per-replicate seeding (seed=None preserves any
    # seeding already done by the caller, e.g. train_and_save).
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    _seed_tag = int(seed) if seed is not None else 0
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

    # task#192 Phase A: removed W_inA_inh / W_inV_inh prints (attributes
    # deleted; direct FF-inh shortcut rip'd).
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
        net.gNMDA = float(os.environ.get("GNMDA", 1.30))   # #66: env-exposed (default 1.30 == byte-identical)
        net.tau_nmda = float(os.environ.get("TAU_NMDA", "80.0"))   # NR2A TAU_NMDA env (default 80.0 == byte-identical)
        net.tau_nmda_v = float(os.environ.get("TAU_NMDA_V", net.tau_nmda))   # latency #153: V-specific FF-NMDA decay (default == net.tau_nmda => byte-identical; > tau_nmda => longer V sub-threshold window)
        net.nmda_alpha = 0.1
        net.Erev_nmda = 20.0
        net.tau_nmdaVolt = 100.0
        net.v_nmda_rest = -65.0
        net.nmda_vrest_offset = 7.0
        net.mg_vhalf = -35.0
        # task#21/E8 (RUN CONFIG §0) — the ONLY parameter change from 3a29fa88.
        # tau_nmda_inh default is 45.0 (interneuron NMDA decay @~1507); 21.6 ms
        # halves the IN NMDA-plateau (Task #18). Smoke asserts ==21.6 @ ep0.
        net.tau_nmda_inh = 21.6


    net.u_a.fill_(0.7)
    net.u_v.fill_(0.7)
    net.tau_rec = 400.0

    print("[tune_for_biology] coarse biological calibration applied")

    # task #106 (researcher #105): manual input_scaling=400 — paired with the
    # tighter target_mean=0.00006 on the MSI-input AMPA/NMDA weights.
    # task #192 Phase A: dropped `net.g_FFinh = 0.6` (orphan attribute now;
    # direct FF-inh shortcut rip'd, so the value has no functional effect).
    net.input_scaling = 400
    net.g_GABA = 10



    # Unsupervised STDP - Useless, no need
    print("\n--- STDP training (unsupervised) ---")
    unsup_start = time.time()
    last_ep = 0
    if panel:
        # task#21/E6a: enable the read-only per-epoch logging panel.
        net._panel_enabled = True
        net._panel_seed = _seed_tag
        net._panel_rows = []
        print(f"[panel] ENABLED seed={_seed_tag} out_dir={panel_out_dir} "
              f"tau_nmda_inh={getattr(net, 'tau_nmda_inh', None)}")
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

        if panel:
            # task#21/E6b: Tap-A taps — FF dW-flux accumulator + in-training E/I
            # recording around train_unsupervised_batch (both read-only).
            net._panel_dW_accum = {nm: 0.0 for nm in
                ('W_inA', 'W_inV', 'W_a2msi_AMPA', 'W_v2msi_AMPA',
                 'W_a2msi_NMDA', 'W_v2msi_NMDA')}
            net.start_ei_recording()

        # Increment 3: recurrent MSI->MSI excitation schedule — OFF until the
        # post-ep25 MSI plasticity stage, then static g_rec=0.1, matching the
        # epoch_idx>25 gate on the recurrent STDP (0-indexed: ON from epoch 26).
        net.g_rec = (float(os.environ.get("G_REC", "0.1")) if epoch > 25 else 0.0)   # G_REC env (default 0.1 == byte-identical)
        net.train_unsupervised_batch(1000, batch_size=256, debug=False, epoch_idx=epoch)  # run some sequences
        net.print_epoch_spike_summary(f"unsup {epoch + 1:02d}")

        if 2 <= epoch <= 79:
            net._probe.report(net, f"epoch {epoch}")

        if panel:
            # task#21/E6c: assemble + append the panel row; full ckpt every-5.
            net._epoch_panel_dump(epoch, _seed_tag)
            if epoch % 5 == 0:
                import os as _os
                _ck = make_checkpoint(net, epoch=epoch,
                                      comment=f"panel seed{_seed_tag} ep{epoch}")
                torch.save(_ck, _os.path.join(panel_out_dir,
                                              f"ckpt_ep{epoch}_seed{_seed_tag}.pt"))

        with torch.no_grad():
            delta = (net.W_inA - W_before).abs().max().item()
            print("Δ‖W_inA‖ =", delta)
        epoch_time = time.time() - epoch_start
        print(f"  Unsup Epoch {epoch + 1}/{n_unsup_epochs} - Time: {epoch_time:.2f}s")
    unsup_time = time.time() - unsup_start
    print(f"Unsupervised training completed in {unsup_time:.2f}s")

    if panel:
        # task#21/E6d: write per-epoch panel rows for this replicate.
        _write_panel_outputs(net._panel_rows, panel_out_dir, _seed_tag)

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

