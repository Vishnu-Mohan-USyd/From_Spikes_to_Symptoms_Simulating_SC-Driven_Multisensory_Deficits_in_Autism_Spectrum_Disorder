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
import copy
import random
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


# ======================================================================
# #62 LEVER L6 — complete state snapshot / restore (verbatim port of the
# proven run34/panelgraph capture machinery). Used ONLY by the L6 capture
# path to make the graph capture byte-TRANSPARENT: the warmup forward steps
# mutate weights+state+RNG; we snapshot everything before capture and
# restore it (in-place copy_, so the persistent state addresses the graph
# baked are preserved) afterwards, so real training replays frame-0
# byte-identically to the eager path. Inert unless FSTS_LEVER_L6=1.
# ======================================================================
def _l6_complete_keys(net):
    """Every mutable float tensor on the net: attrs + params + float buffers."""
    keys = []
    for k, v in list(vars(net).items()):
        if torch.is_tensor(v) and v.is_floating_point():
            keys.append(("attr", k))
    for k, _v in net.named_parameters():
        keys.append(("param", k))
    for k, v in net.named_buffers():
        if v is not None and torch.is_tensor(v) and v.is_floating_point():
            keys.append(("buf", k))
    return keys


def _l6_resolve(net, kind, name):
    if kind == "attr":
        return getattr(net, name)
    obj = net
    *parents, last = name.split(".")
    for p in parents:
        obj = getattr(obj, p)
    return getattr(obj, last)


def _l6_snapshot(net, keys):
    snap, seen = {}, set()
    for kind, name in keys:
        t = _l6_resolve(net, kind, name)
        if not torch.is_tensor(t) or id(t) in seen:
            continue
        seen.add(id(t)); snap[(kind, name)] = t.detach().clone()
    snap[("_pos", "_delay_positions")] = copy.deepcopy(net._delay_positions)
    snap[("_gpos", "g")] = {s: getattr(net, '_gpos_' + s).clone()
                            for _b, s in net._GPOS_NAMES
                            if torch.is_tensor(getattr(net, '_gpos_' + s, None))}
    return snap


def _l6_restore(net, snap):
    net._delay_positions = copy.deepcopy(snap[("_pos", "_delay_positions")])
    for s, v in snap[("_gpos", "g")].items():
        getattr(net, '_gpos_' + s).copy_(v)
    for (kind, name), val in snap.items():
        if kind in ("_pos", "_gpos"):
            continue
        t = _l6_resolve(net, kind, name)
        if torch.is_tensor(t) and t.shape == val.shape:
            t.copy_(val)


def _l6_snapshot_rng():
    return (torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            np.random.get_state(), random.getstate())


def _l6_restore_rng(s):
    torch.set_rng_state(s[0])
    if s[1] is not None:
        torch.cuda.set_rng_state_all(s[1])
    np.random.set_state(s[2]); random.setstate(s[3])


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
        # ── Perf speed levers (task#33; debugger-proven dbg_perf_20260613).
        #    Individually toggleable + env-overridable (FSTS_LEVER_L{1,2,3}).
        #    L1/L2 are bit-exact identity-preserving (cheap GATEs 1 & 2 == 0.0) and
        #    DEFAULT ON — this is the accepted ship build. L3 is per-step reduction-
        #    order-equivalent (~6e-8) but ACCUMULATES over epochs (KILLs GATE-2
        #    strict identity), so it is DROPPED: DEFAULT OFF, opt-in only via
        #    FSTS_LEVER_L3=1 (reserved for a future reduction-order-fixed / lever-4
        #    path). Each lever's OFF branch is the VERBATIM canonical code, so
        #    all-levers-OFF reproduces canonical Training.py (md5 466c9a76) bit-for-bit.
        #    L1 = skip the in-loop EI/dW-flux .item() host-sync logging during the
        #         main training epoch (keep the every-5 panel-battery summaries).
        #    L2 = vectorize the 5 boolean-masked Izhikevich recovery resets.
        #    L3 = STDP per-sample torch.ger loop -> single matmul accumulation.
        self._lever_L1_drop_inloop_log = (os.environ.get('FSTS_LEVER_L1', '1') == '1')
        self._lever_L2_vector_reset    = (os.environ.get('FSTS_LEVER_L2', '1') == '1')
        self._lever_L3_stdp_matmul     = (os.environ.get('FSTS_LEVER_L3', '0') == '1')  # DROPPED -> default OFF
        #    L5 = alloc-free reset_state (#61): restore per-mini-batch init values IN-PLACE
        #         on address-stable PERSISTENT pools (TASK #34) instead of reallocating new
        #         state tensors each reset. DECOUPLED from L4's gpos forward path so the
        #         reset alone can be single-variable byte-identity-gated. OFF branch = the
        #         verbatim realloc reset (existing default); the train-graph (#62) needs ON.
        self._lever_L5_allocfree_reset = (os.environ.get('FSTS_LEVER_L5', '0') == '1')
        #    L6 = CUDA-graph the train substep loop (#62): per external step, REPLAY the
        #         captured 100-substep update_all_layers_batch (plasticity ON — the in-loop
        #         topographic anchor every 10 substeps is INSIDE the graph; the post-loop
        #         competition/soft-scaling and ALL STDP stay EAGER, per ext-step). Weights are
        #         live nn.Parameters (in-place _p_add), so the captured graph reads/writes
        #         their static addresses every replay; only the per-substep INPUT (xA/xV/valid)
        #         needs static staging buffers. L6 ON forces L4 ON (needs the alloc-free reset
        #         + gpos cursors + _g_out_* static-address outputs). OFF branch = the verbatim
        #         eager substep loop (byte-identical). DEFAULT OFF (env FSTS_LEVER_L6).
        self._lever_L6_graph_train     = (os.environ.get('FSTS_LEVER_L6', '0') == '1')
        if self._lever_L6_graph_train:
            self._lever_L4_graph = True        # L6 requires the L4 graphing substrate
        #    L7 = CUDA-graph the per-ext-step STDP ger-sequence (the eager tail OUTSIDE the L6
        #         forward graph). Captures the exact sequential gers (NOT a matmul) -> removes
        #         CPU dispatch of the ~1024 tiny ger kernels while preserving FP order ->
        #         bit-identical. Default OFF; single-variable. L7 => L6 => L4.
        self._lever_L7_graph_stdp      = (os.environ.get('FSTS_LEVER_L7', '0') == '1')
        if self._lever_L7_graph_stdp:
            self._lever_L6_graph_train = True  # L7 graphs the STDP tail ON TOP of the L6 forward
            self._lever_L4_graph       = True  # ... which needs the L4 graphing substrate
        #    L8 = FUSE the whole per-ext-step body — forward substeps + post-loop plasticity
        #         (anchors/competition/soft-scaling) + the STDP tail — into ONE capture-last CUDA
        #         graph (#69). Eliminates BOTH the eager STDP tail AND the #68 two-graph hazard
        #         (L7's separate g_stdp). MUTUALLY EXCLUSIVE with L6/L7 (never co-capture two
        #         graphs). Forces the L4 substrate. The Poisson presyn spikes stay eager — sampled
        #         BEFORE the atomic replay and staged into static buffers the captured tail reads;
        #         the forward is RNG-free, so that reorder is byte-identical. DEFAULT OFF (FSTS_LEVER_L8).
        self._lever_L8_fused           = (os.environ.get('FSTS_LEVER_L8', '0') == '1')
        # #74 PATH B: L8-FULL — fold the unimodal lateral-competition matmul (and everything after it)
        # INTO the single graph too = the maximal fusion the ladder timed at ~5.89-min/80ep. NOT
        # byte-identical: the in-graph GEMM diverges from eager by ~1.58e-4 on W_inA/W_inV (debugger
        # #73). This is the user-authorized relax for the <=10-min wall, gated downstream by a
        # TBW/SBW/E-I equivalence re-val. OFF => L8 stays the byte-identical PATH-A partial fusion.
        self._lever_L8_full            = (os.environ.get('FSTS_LEVER_L8_FULL', '0') == '1')
        if self._lever_L8_full:
            self._lever_L8_fused       = True   # L8-full is a MODE of L8 -> force the single L8 graph
        if self._lever_L8_fused:
            self._lever_L4_graph       = True   # L8 needs the L4 substrate (_g_out_*/gpos/alloc-free reset)
            self._lever_L6_graph_train = False  # L8 REPLACES the L6 forward graph ...
            self._lever_L7_graph_stdp  = False  # ... and subsumes the L7 STDP-tail graph (single graph only)

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
        self._prev_sMSI_rec = None  # REVERT(debugger#60): restore delay-fix EDIT#2 state field
        self.tau_istdp_pre = 20.0    # ms — symmetric pre-trace decay
        self.tau_istdp_post = 20.0   # ms — symmetric post-trace decay
        self.eta_istdp = 2e-5        # was 5e-4 (25× lower); debugger #193 §Q3.E range 1e-5 to 5e-5
        self.istdp_baseline = 0.6    # was 0.0 (bug); trace_ss at ρ=30 Hz × τ_pre = 30·0.02 = 0.6
                                     # biology: Stein-Stanford 2008 cat SC sustained 10-30 Hz
        self.W_gaba_clamp = 0.5      # upper bound on W_msiInh2Exc_GABA per spec

        # ------------- Izhikevich params --------------
        # For unimodal excit
        self.aA, self.bA, self.cA, self.dA = 0.02, 0.2, -65.0, 8.0
        self.aV, self.bV, self.cV, self.dV = 0.02, 0.2, -65.0, 8.0
        # MSI excit
        self.aM, self.bM, self.cM, self.dM = 0.02, 0.2, -65.0, 8.0
        # MSI inh (fast spiking)
        self.aMi, self.bMi, self.cMi, self.dMi = 0.1, 0.2, -65.0, 2.0
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
        self.tau_gaba = 50.0   # ms
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
        self.conduction_delay_v2msi = 400   # 40.0 ms visual, slower-arriving => V-leading-wider TBW
        self.conduction_delay_msi2out = conduction_delay_msi2out
        self.conduction_delay_msi_rec = 100   # delay-fix: 100 substeps = 10.0 ms = 1 ext step

        # task #192 Phase A: REMOVED conduction_delay_inA_inh / _inV_inh
        # (paired with the direct FF-inh shortcut rip in __init__ above).

        # unimodal->MSI_inh excit:
        # task #192 Phase C: 4 ms (same as exc; one EPSC leg onto interneuron).
        self.conduction_delay_a2msi_inh = 270   # = a2msi + 20 (2.0 ms exc leg onto interneuron)
        self.conduction_delay_v2msi_inh = 420   # = v2msi + 20

        # MSI_inh->MSI_exc
        # task #192 Phase C: 5.2 ms over the exc->inh leg, totalling
        # disynaptic ~9.2 ms (Whyland-Bickford 2018 IPSC).
        self.conduction_delay_msi_inh2exc = 450   # = max(a2msi,v2msi) + 50 (disynaptic IPSC)

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
        self.tau_nmda_inh = 45.0  # task#14 route-c TBW fix: GluN2A-fast PV value (adult fast-spiking PV is GluN2A-dominated ~30-45ms). Was 25.0 (route-C TBW-tune); 90ms is a GluN2B miscite from rat CA1.
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
        self._ensure_delay_buffer("buffer_a2msi",       delay=d_a2msi,       width=self.n)
        self._ensure_delay_buffer("buffer_v2msi",       delay=d_v2msi,       width=self.n)
        self._ensure_delay_buffer("buffer_a2msi_inh",   delay=d_a2msi_inh,   width=self.n)
        self._ensure_delay_buffer("buffer_v2msi_inh",   delay=d_v2msi_inh,   width=self.n)
        self._ensure_delay_buffer("buffer_msi_inh2exc", delay=d_msi_inh2exc, width=self.n_inh)
        self._ensure_delay_buffer("buffer_msi2out",     delay=d_msi2out,     width=self.n)
        self._ensure_delay_buffer("buffer_msi_rec",     delay=d_msi_rec,     width=self.n)

    # ---- LEVER L4 (CUDA-graph) ring-buffer cursors --------------------------
    # The python-int ring positions in `_delay_positions` get BAKED into a CUDA
    # graph (the index `buf[pos]` is a python int at trace time), so a captured
    # graph is only correct for its exact frame. To make ONE graph replay across
    # frames bit-identically, the cursors become persistent GPU long tensors that
    # are READ at execution time (index_select/index_copy_) and advanced IN-GRAPH
    # (add_/remainder_). create-or-zero-IN-PLACE so their addresses stay static
    # across replays (CUDA-graph requirement). Numerically identical: same rows,
    # same integer modular arithmetic. Only used when `_lever_L4_graph` is set.
    _GPOS_NAMES = [("buffer_a2msi", "a2msi"), ("buffer_v2msi", "v2msi"),
                   ("buffer_a2msi_inh", "a2msi_inh"), ("buffer_v2msi_inh", "v2msi_inh"),
                   ("buffer_msi_inh2exc", "msi_inh2exc"), ("buffer_msi2out", "msi2out"),
                   ("buffer_msi_rec", "msi_rec")]

    def _sync_graph_pos(self) -> None:
        """Create (once) or in-place re-sync the GPU position cursors from the
        python-int `_delay_positions`. In-place fill keeps addresses static."""
        for bufname, short in self._GPOS_NAMES:
            attr = "_gpos_" + short
            val = int(self._delay_positions[bufname])
            t = getattr(self, attr, None)
            # Compare device TYPE only (cpu vs cuda): self.device is often 'cuda'
            # with no index while the tensor lands on cuda:0, so a strict !=
            # would recreate the cursor every call and break the CUDA-graph
            # (which reads it at a baked address). Create once, then fill IN-PLACE.
            if t is None or t.device.type != torch.device(self.device).type:
                setattr(self, attr, torch.tensor([val], device=self.device, dtype=torch.long))
            else:
                t.fill_(val)

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

        # ---- LEVER L4 (CUDA-graph) graph-safe in-place reset -----------------
        # A captured CUDA graph requires the neuron-state tensors to keep STATIC
        # addresses across replays, but the eager reset below REALLOCATES them
        # every mini-batch. So under _lever_L4_graph we cache the freshly-built
        # init state once (first reset) and on subsequent same-batch resets we
        # restore those exact values IN-PLACE (copy_) — bit-identical values,
        # stable addresses. Gated OFF by default => flag-off path is unchanged.
        # LEVER L5 (decoupled from L4's gpos): the alloc-free pool reset runs under EITHER
        # the full graph lever OR the standalone alloc-free-reset lever, so the reset can be
        # byte-identity-gated WITHOUT enabling the gpos forward path (which stays L4-only).
        _l4 = getattr(self, '_lever_L4_graph', False) or getattr(self, '_lever_L5_allocfree_reset', False)
        # TASK #34: per-batch-size PERSISTENT pools ("two-buffer-set reset"). The captured
        # CUDA graph bakes the B=256 pool's addresses; the canonical 1000-cadence B=232
        # tail runs EAGER on a SEPARATE persistent B=232 pool. Both pools are held alive by
        # these caches and restored IN-PLACE (copy_), so neither is ever freed/realloc'd ->
        # the graph's baked pointers stay valid across the intervening eager tail batch.
        _caches = getattr(self, '_graph_reset_caches', None)
        _gc = _caches.get(self.batch_size) if (_l4 and _caches is not None) else None
        if _gc is not None:
            for nm in _gc['_names']:
                t = _gc['_pool'][nm]          # persistent storage for THIS batch size
                t.copy_(_gc['_init'][nm])      # bit-exact fresh-reset values
                setattr(self, nm, t)           # rebind (attr may point at the other pool)
            self._delay_positions = {k: 0 for k in self._delay_positions}
            self._sync_graph_pos()
            self._dbg_spk_A = self._dbg_spk_V = self._dbg_spk_MSI = 0.0
            self._dbg_steps = 0
            self._dbg_spk_Mi = 0.0
            # faithful replication of the eager reset (L2559): the delayed recurrent pre is
            # a NON-pool field (None at pool-build) so it is NOT restored by the copy_ loop
            # above; clear it here or it leaks across mini-batches at epoch>25 (recurrent STDP).
            self._prev_sMSI_rec = None
            return
        _l4_build = _l4    # no pool yet for this batch size -> full alloc below, then cache
        if _l4_build:
            _pre_ids = {k: id(v) for k, v in vars(self).items()
                        if torch.is_tensor(v) and v.is_floating_point()}

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
        self._prev_sMSI_rec = None  # REVERT(debugger#60): clear delayed recurrent pre at seq start
        self.ampa_m.zero_()  # clear low-pass AMPA state

        # clear conduction ring buffers
        self._reset_delay_buffers()

        if hasattr(self, "ampa_m"):
            self.ampa_m = torch.zeros((self.batch_size, self.n),
                                      dtype=torch.float32,
                                      device=self.device)

        # reset NMDA gating
        self.nmda_m = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
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
        self._dbg_spk_Mi = 0.0  # task#21/S1: PV/FS spike accumulator

        # task #185 (minimal-arch §5.1): REMOVED AGC gate-state reset
        # (_last_agc_fast_t, _last_agc_slow_t no longer exist).

        # reset RF tracking
        self.msi_rf_centers = torch.zeros((self.batch_size, self.n), device=self.device)
        self.msi_rf_certainty = torch.zeros((self.batch_size, self.n), device=self.device)

        # ---- LEVER L4: build the graph-safe reset cache (first reset only) ----
        # Names this reset (re)allocated (id changed or newly created) are exactly
        # the per-mini-batch state set; the plastic weights keep their id (they are
        # untouched here) and so are correctly EXCLUDED — STDP-learned weights are
        # never clobbered by the in-place restore. The 6 conduction ring buffers
        # are unioned in explicitly because _ensure_delay_buffer zeroes them in
        # place (id unchanged) when the shape already matches.
        if _l4_build:
            names = [k for k, v in vars(self).items()
                     if torch.is_tensor(v) and v.is_floating_point()
                     and (k not in _pre_ids or id(v) != _pre_ids[k])]
            for bufname, _short in self._GPOS_NAMES:
                if bufname not in names and torch.is_tensor(getattr(self, bufname, None)):
                    names.append(bufname)
            # PERSISTENT pool = the just-allocated live tensors (held alive by this cache,
            # so the caching allocator never reuses their blocks); _init = bit-exact reset
            # values restored in-place on every subsequent same-B reset.
            pool = {nm: getattr(self, nm) for nm in names}
            init = {nm: getattr(self, nm).detach().clone() for nm in names}
            if getattr(self, '_graph_reset_caches', None) is None:
                self._graph_reset_caches = {}
            self._graph_reset_caches[self.batch_size] = {
                '_names': names, '_pool': pool, '_init': init}
            self._sync_graph_pos()

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
        W_a2msiInh_AMPA = self.W_a2msiInh_AMPA
        W_a2msiInh_NMDA = self.W_a2msiInh_NMDA
        W_v2msiInh_AMPA = self.W_v2msiInh_AMPA
        W_v2msiInh_NMDA = self.W_v2msiInh_NMDA
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
        buf_msi_rec = self.buffer_msi_rec   # delay-fix

        pos_a2msi = self._delay_positions["buffer_a2msi"]
        pos_v2msi = self._delay_positions["buffer_v2msi"]
        # task #192 Phase A: REMOVED pos_inA_inh / pos_inV_inh ring-buffer cursors.
        pos_a2msi_inh = self._delay_positions["buffer_a2msi_inh"]
        pos_v2msi_inh = self._delay_positions["buffer_v2msi_inh"]
        pos_msi_inh2exc = self._delay_positions["buffer_msi_inh2exc"]
        pos_msi2out = self._delay_positions["buffer_msi2out"]
        pos_msi_rec = self._delay_positions["buffer_msi_rec"]   # delay-fix

        # LEVER L4 (CUDA-graph): use persistent GPU cursors so the ring-buffer
        # indices are read/advanced at execution time (frame-reusable graph).
        # OFF by default -> the python-int path below is byte-for-byte the ship code.
        _l4 = getattr(self, '_lever_L4_graph', False)
        if _l4:
            gpos_a2msi = self._gpos_a2msi
            gpos_v2msi = self._gpos_v2msi
            gpos_a2msi_inh = self._gpos_a2msi_inh
            gpos_v2msi_inh = self._gpos_v2msi_inh
            gpos_msi_inh2exc = self._gpos_msi_inh2exc
            gpos_msi2out = self._gpos_msi2out
            gpos_msi_rec = self._gpos_msi_rec

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
        # LEVER-4 capture-safety: torch.tensor(scalar, device=cuda) does a host->device
        # copy that ABORTS CUDA-graph capture (cudaErrorStreamCaptureInvalidated).
        # torch.zeros((), ...) is a memset kernel = capture-safe AND value-identical (0.0).
        dbg_spk_A = torch.zeros((), device=self.device)
        dbg_spk_V = torch.zeros((), device=self.device)
        dbg_spk_Mi = torch.zeros((), device=self.device)  # task#21/S1: PV/FS
        dbg_spk_M = torch.zeros((), device=self.device)

        for sub_i in range(self.n_substeps):
            # LEVER-4 capture diagnostic: cap the substep count (off by default →
            # no change). Used ONLY to bisect which substep op invalidates capture;
            # does NOT touch the n_substeps hyperparameter or any real run.
            if getattr(self, '_cap_max_substeps', None) is not None and sub_i >= self._cap_max_substeps:
                break
            # --- Decay old currents ---
            # task #192 Phase B: I_M_gaba decays at tau_gaba (slow);
            # I_M (AMPA + NMDA-driven) decays at tau_ampa (fast).
            self.I_A.mul_(decay_factor)
            self.I_V.mul_(decay_factor)
            self.I_M.mul_(decay_factor)
            self.I_M_gaba.mul_(gaba_decay)
            self.I_M_inh.mul_(decay_factor)
            self.I_O.mul_(decay_factor)

            # Add external input (split across substeps)
            self.I_A.add_(I_A_input * input_step_scale)
            self.I_V.add_(I_V_input * input_step_scale)

            # --- Ring-buffer reads (replaces deque popleft) ---
            # task #192 Phase A: REMOVED delayed_spikes_inA_inh / _inV_inh
            # reads (direct FF-inh shortcut rip'd).
            if _l4:  # LEVER L4: GPU-cursor gather (index read at exec time)
                delayed_spikes_a2msi = buf_a2msi.index_select(0, gpos_a2msi).squeeze(0) if delay_a2msi > 0 else zero_exc
                delayed_spikes_v2msi = buf_v2msi.index_select(0, gpos_v2msi).squeeze(0) if delay_v2msi > 0 else zero_exc
                delayed_spikes_a2msi_inh = buf_a2msi_inh.index_select(0, gpos_a2msi_inh).squeeze(0) if delay_a2msi_inh > 0 else zero_exc
                delayed_spikes_v2msi_inh = buf_v2msi_inh.index_select(0, gpos_v2msi_inh).squeeze(0) if delay_v2msi_inh > 0 else zero_exc
                delayed_spikes_msi_inh2exc = buf_msi_inh2exc.index_select(0, gpos_msi_inh2exc).squeeze(0) if delay_msi_inh2exc > 0 else zero_inh
                delayed_spikes_msi2out = buf_msi2out.index_select(0, gpos_msi2out).squeeze(0) if delay_msi2out > 0 else zero_exc
                delayed_spikes_msi_rec = buf_msi_rec.index_select(0, gpos_msi_rec).squeeze(0) if delay_msi_rec > 0 else zero_exc
            else:
                delayed_spikes_a2msi = buf_a2msi[pos_a2msi] if delay_a2msi > 0 else zero_exc
                delayed_spikes_v2msi = buf_v2msi[pos_v2msi] if delay_v2msi > 0 else zero_exc
                delayed_spikes_a2msi_inh = buf_a2msi_inh[pos_a2msi_inh] if delay_a2msi_inh > 0 else zero_exc
                delayed_spikes_v2msi_inh = buf_v2msi_inh[pos_v2msi_inh] if delay_v2msi_inh > 0 else zero_exc
                delayed_spikes_msi_inh2exc = buf_msi_inh2exc[pos_msi_inh2exc] if delay_msi_inh2exc > 0 else zero_inh
                delayed_spikes_msi2out = buf_msi2out[pos_msi2out] if delay_msi2out > 0 else zero_exc
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
            mg_iA = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_inhA - self.mg_vhalf)))
            mg_iV = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_inhV - self.mg_vhalf)))
            # task#21/E2/S2: INH-plateau accumulator — battery single-volley probe ONLY
            # (_panel_inh_accum is None during training => strict no-op, trajectory-neutral)
            if self._panel_inh_accum is not None:
                _rg = self._rec_gpu   # delay-fix/graph: capture-safe per-substep buffer
                _rg['mg_iA'][sub_i] = mg_iA.mean()
                _rg['mg_iV'][sub_i] = mg_iV.mean()
                _rg['mg_iA_on'][sub_i] = (mg_iA > 0.5).float().mean()
                _rg['mg_iV_on'][sub_i] = (mg_iV > 0.5).float().mean()
                _rg['v_dend_inhA'][sub_i] = self.v_dend_inhA.mean()
                _rg['v_dend_inhV'][sub_i] = self.v_dend_inhV.mean()
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
            self.I_M_gaba.add_(I_M_inh2exc)

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
            if self._lever_L2_vector_reset:    # LEVER L2: vectorized reset (kills aten::nonzero host-sync)
                self.u_uniA += new_sA * self.dA
            else:
                self.u_uniA[spike_mask_A] += self.dA

            # V
            dVV = (0.04 * self.v_uniV.pow(2) + 5.0 * self.v_uniV + 140.0
                   - self.u_uniV + self.I_V)
            self.v_uniV += self.dt * dVV
            self.u_uniV += self.dt * (self.aV * (self.bV * self.v_uniV - self.u_uniV))
            spike_mask_V = (self.v_uniV >= spike_threshold)
            new_sV = spike_mask_V.float()
            self.v_uniV.masked_fill_(spike_mask_V, self.cV)
            if self._lever_L2_vector_reset:    # LEVER L2: vectorized reset
                self.u_uniV += new_sV * self.dV
            else:
                self.u_uniV[spike_mask_V] += self.dV

            # MSI excit
            # task #192 Phase B: net current is (I_M - I_M_gaba) where I_M
            # holds AMPA/NMDA-driven fast excitation (decays at tau_ampa=2.5ms)
            # and I_M_gaba holds tonic + disynaptic + surround GABAergic
            # inhibition magnitude (decays at tau_gaba=50ms slow IPSC biology).
            dVM = (0.04 * self.v_msi.pow(2) + 5.0 * self.v_msi + 140.0
                   - self.u_msi + (self.I_M - self.I_M_gaba))
            self.v_msi += self.dt * dVM
            self.u_msi += self.dt * (self.aM * (self.bM * self.v_msi - self.u_msi))
            spike_mask_M = (self.v_msi >= spike_threshold)
            new_sM = spike_mask_M.float()
            self.v_msi.masked_fill_(spike_mask_M, self.cM)
            if self._lever_L2_vector_reset:    # LEVER L2: vectorized reset
                self.u_msi += new_sM * self.dM
            else:
                self.u_msi[spike_mask_M] += self.dM

            # (A) compute surround inhibition current
            I_latM = torch.mm(new_sM, self.W_MSI_inh)  # shape (B, n)
            # (B) apply it
            # task #192 Phase B: route Mexican-hat lateral GABA into I_M_gaba
            # (tau_gaba decay = 50ms biology) instead of fast I_M. Magnitude
            # is added as positive; subtracted from I_M at integration.
            self.I_M_gaba.add_(self.g_GABA * I_latM)

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
                g_rec_syn = F.linear(delayed_spikes_msi_rec, self.W_MSI_exc)            # (B,n) conductance
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
                _I_I_lat = torch.clamp(self.g_GABA * I_latM, min=0.0)
                _I_I = _I_I_ff + _I_I_recur + _I_I_lat

                _rg = self._rec_gpu   # delay-fix/graph: capture-safe per-substep buffer
                _rg["I_E_mean"][sub_i] = _I_E.mean()
                _rg["I_I_mean"][sub_i] = _I_I.mean()
                _rg["Q_E"][sub_i] = (_I_E * dt_s).mean()
                _rg["Q_I"][sub_i] = (_I_I * dt_s).mean()
                _rg["AMPA"][sub_i] = _I_E_ampa.mean()
                _rg["NMDA"][sub_i] = _I_E_nmda.mean()
                _rg["FFInh"][sub_i] = _I_I_ff.mean()
                _rg["RecurInh"][sub_i] = _I_I_recur.mean()
                _rg["LatInh"][sub_i] = _I_I_lat.mean()

            if self.enable_probe and self._probe is not None:
                # task #192 Phase B: I_total is the NET current onto MSI exc,
                # i.e. fast (AMPA + NMDA + tonic stays-in-I_M dropped) minus
                # slow GABA state. Mirrors what Izhikevich actually integrates.
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
            if self._lever_L2_vector_reset:    # LEVER L2: vectorized reset
                self.u_msi_inh += new_sMi * self.dMi
            else:
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
            if self._lever_L2_vector_reset:    # LEVER L2: vectorized reset
                self.u_out += new_sO * self.dO
            else:
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

            # Ring-buffer writes (replaces deque append)
            # task #192 Phase A: REMOVED inA_inh / inV_inh writes.
            if _l4:  # LEVER L4: GPU-cursor scatter + in-graph modular advance
                if delay_a2msi > 0:
                    buf_a2msi.index_copy_(0, gpos_a2msi, new_sA.unsqueeze(0))
                    gpos_a2msi.add_(1).remainder_(delay_a2msi)
                    # SHIP-PARITY (bit-identity): the python-int read `buf[pos]` is a
                    # VIEW that this same-row write OVERWRITES, so the ship's RETURNED
                    # `delayed_spikes_a2msi` == new_sA (the current spike), NOT the
                    # delayed value. The `index_select` read above is a COPY (true
                    # delayed) — correct for the IN-LOOP A->MSI drive (consumed before
                    # this write, exactly as the ship consumes its pre-write view), but
                    # it leaves the RETURNED / _latest_dA2M value as the true-delayed
                    # spike, which feeds the A->MSI STDP a different pre-spike than the
                    # ship -> weight drift -> I_M divergence. Re-bind to reproduce the
                    # ship's view-clobber for the returned/_latest value only.
                    delayed_spikes_a2msi = new_sA
                if delay_v2msi > 0:
                    buf_v2msi.index_copy_(0, gpos_v2msi, new_sV.unsqueeze(0))
                    gpos_v2msi.add_(1).remainder_(delay_v2msi)
                    delayed_spikes_v2msi = new_sV  # SHIP-PARITY: see a2msi note above
                if delay_a2msi_inh > 0:
                    buf_a2msi_inh.index_copy_(0, gpos_a2msi_inh, new_sA.unsqueeze(0))
                    gpos_a2msi_inh.add_(1).remainder_(delay_a2msi_inh)
                if delay_v2msi_inh > 0:
                    buf_v2msi_inh.index_copy_(0, gpos_v2msi_inh, new_sV.unsqueeze(0))
                    gpos_v2msi_inh.add_(1).remainder_(delay_v2msi_inh)
                if delay_msi_inh2exc > 0:
                    buf_msi_inh2exc.index_copy_(0, gpos_msi_inh2exc, new_sMi.unsqueeze(0))
                    gpos_msi_inh2exc.add_(1).remainder_(delay_msi_inh2exc)
                if delay_msi2out > 0:
                    buf_msi2out.index_copy_(0, gpos_msi2out, new_sM.unsqueeze(0))
                    gpos_msi2out.add_(1).remainder_(delay_msi2out)
                if delay_msi_rec > 0:
                    buf_msi_rec.index_copy_(0, gpos_msi_rec, new_sM.unsqueeze(0))
                    gpos_msi_rec.add_(1).remainder_(delay_msi_rec)
            else:
                if delay_a2msi > 0:
                    buf_a2msi[pos_a2msi].copy_(new_sA)
                    pos_a2msi = (pos_a2msi + 1) % delay_a2msi
                if delay_v2msi > 0:
                    buf_v2msi[pos_v2msi].copy_(new_sV)
                    pos_v2msi = (pos_v2msi + 1) % delay_v2msi
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
                if delay_msi_rec > 0:
                    buf_msi_rec[pos_msi_rec].copy_(new_sM)
                    pos_msi_rec = (pos_msi_rec + 1) % delay_msi_rec

            sA, sV, sM, sMi, sO = new_sA, new_sV, new_sM, new_sMi, new_sO

            self._latest_sA = sA
            self._latest_sV = sV
            self._latest_sMSI = sM
            self._latest_sMSI_inh = sMi
            if _l4:  # LEVER L4: expose last-substep delayed spikes (dA2M/dV2M) as
                # persistent attrs so the eager STDP tail can read them post-replay
                # (the captured graph returns None; these rebind to graph statics).
                self._latest_dA2M = delayed_spikes_a2msi
                self._latest_dV2M = delayed_spikes_v2msi

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
        # LEVER L4: skip — the GPU cursors (_gpos_*) were advanced in-graph and are
        # the source of truth; writing the un-advanced python ints would corrupt them.
        if not _l4:
            self._delay_positions["buffer_a2msi"] = pos_a2msi
            self._delay_positions["buffer_v2msi"] = pos_v2msi
            self._delay_positions["buffer_a2msi_inh"] = pos_a2msi_inh
            self._delay_positions["buffer_v2msi_inh"] = pos_v2msi_inh
            self._delay_positions["buffer_msi_inh2exc"] = pos_msi_inh2exc
            self._delay_positions["buffer_msi2out"] = pos_msi2out
            self._delay_positions["buffer_msi_rec"] = pos_msi_rec

        # LEVER L4: write the final-substep spikes into PERSISTENT, pre-allocated
        # output buffers so the wrapper reads STATIC addresses. Tensors that ESCAPE a
        # capture via `self._latest_* = sA` are unsafe once a SECOND phase graph is
        # captured — its pool reuse stales the first (le25) graph's escaped outputs,
        # so the wrapper silently read corrupted spikes (GATE 1/2 never hit this: one
        # graph only). Copying into persistent buffers shared by ALL phase graphs
        # (in-place) removes the escape. Gated on the buffers existing (allocated by
        # the lever-4 driver) → strict no-op for eager/canonical training.
        # Only the B=256 graph batches write these (buffers are sized to the captured B);
        # the eager B=232 tail returns its real spikes directly, so skip on a shape
        # mismatch instead of erroring.
        if (_l4 and getattr(self, '_g_out_sA', None) is not None
                and sA.shape[0] == self._g_out_sA.shape[0]):
            self._g_out_sA.copy_(sA)
            self._g_out_sV.copy_(sV)
            self._g_out_sMSI.copy_(sM)
            self._g_out_sMSI_inh.copy_(sMi)   # #66: persist escaped inh spikes (B, n_inh)
            self._g_out_dA2M.copy_(delayed_spikes_a2msi)
            self._g_out_dV2M.copy_(delayed_spikes_v2msi)
            if return_spike_sum and getattr(self, '_g_out_sumM', None) is not None:
                self._g_out_sumM.copy_(sum_sM)

        # LEVER-4 probe hook: stop right after the 100-substep loop, before the
        # post-loop plasticity (apply_local_competition_*/soft_row_scaling use
        # .median()/.norm() which may not be stream-capture-safe) — lets us test
        # whether the SUBSTEP LOOP ITSELF captures, isolating any capture blocker.
        # OFF by default → zero behaviour change to normal training.
        if getattr(self, '_cap_stop_after_substeps', False):
            return None

        # ------ sync debug counters (one GPU->CPU transfer) ------
        # LEVER-4 probe hook: this .item() block is the only host-sync left on the
        # training path inside update_all_layers_batch; skipping it lets the whole
        # method be CUDA-graph-captured. Counters are non-functional bookkeeping
        # (do not affect weights/dynamics). Flag is OFF by default → zero behaviour
        # change to normal training; only the capture harness sets it True.
        if not getattr(self, '_cap_skip_dbg_sync', False):
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

            # LEVER L8 (#74) capture hook: stop AFTER the post-loop anchor, BEFORE the unimodal
            # competition. The competition's neighbour-mask matmul (s=matmul(M,r), L520) binds a
            # different GEMM kernel under CUDA-graph capture than eager → W_in byte-divergence of
            # 1.58e-4 (debugger #73, proven by single-variable causal flip). So the L8 fused graph
            # captures substeps + in-loop + post-loop anchor only; competition/msi/soft_row and the
            # STDP tail run EAGER post-replay (mirrors L6's certified discipline). OFF by default →
            # strict no-op for eager/L6 training.
            if getattr(self, '_cap_stop_after_anchor', False):
                return None

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
            if self._lever_L3_stdp_matmul:
                # LEVER L3: sum_b ger(post_spk[b], pre_trace[b]) == post_spk.T @ pre_trace
                ltp = post_spk.t() @ pre_trace
                ltd = post_trace.t() @ pre_spk
            else:
                ltp = torch.zeros_like(W)
                ltd = torch.zeros_like(W)
                for b in range(B):
                    ltp += torch.ger(post_spk[b], pre_trace[b])
                    ltd += torch.ger(post_trace[b], pre_spk[b])
            # soft-bound per-element LTP/LTD gates stay OUTSIDE the accumulation (unchanged)
            _g_ltp = (_Wmax - W).clamp(min=0.0).pow(_mu)
            _g_ltd = W.clamp(min=0.0).pow(_mu)
            dW = A_plus * _g_ltp * ltp - A_minus * _g_ltd * ltd
        else:
            if self._lever_L3_stdp_matmul:
                # LEVER L3: same matmul identity (non-soft-bound path: W_inA/V, W_MSI_exc)
                dW = A_plus * (post_spk.t() @ pre_trace) - A_minus * (post_trace.t() @ pre_spk)
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

    # ─────────────────────────────────────────────────────────
    # #62 LEVER L6 — capture & replay the train substep graph
    # ─────────────────────────────────────────────────────────
    def _l6_capture_phase(self, B, epoch_idx, g_rec):
        """Capture the LOOP-ONLY substep graph for one phase (plasticity ON, so the
        in-loop topographic anchor is baked into the graph). RNG- and weight-DIRTY
        (samples its own warmup inputs, advances weights via the full-fwd warmup); the
        caller (_l6_install) wraps this in a complete snapshot/restore so the net is left
        unchanged. Returns (graph, sxA, sxV, svm) — the static input staging buffers the
        replay wrapper copy_'s into each ext-step. Mirrors the proven panel capture_phase."""
        self.g_rec = g_rec
        loc, mod, ok, lens = generate_event_loc_seq_batch(
            batch_size=B, space_size=self.space_size,
            offset_probability=0.6, temporal_jitter_max=2)
        T = max(lens)
        inten = 10 ** torch.empty(1).uniform_(math.log10(0.4), 0).item()
        xA, xV, valid = generate_av_batch_tensor(
            loc, mod, ok, n=self.n, space_size=self.space_size, sigma_in=self.sigma_in,
            noise_std=self.noise_std, loc_jitter_std=self.loc_jitter_std,
            stimulus_intensity=inten, device=self.device, max_len=T)

        self.reset_state(B)
        self._cap_skip_dbg_sync = True
        K = min(6, T - 1)
        self._cap_stop_after_substeps = False          # warm FULL fwd -> builds kernel caches
        for t in range(K):
            self.update_all_layers_batch(xA[:, t], xV[:, t], valid[:, t],
                                         epoch_idx=epoch_idx, return_delayed=True)
        sxA = xA[:, K].contiguous().clone()
        sxV = xV[:, K].contiguous().clone()
        svm = valid[:, K].contiguous().clone()

        self._cap_stop_after_substeps = True           # *** capture LOOP ONLY ***
        def call_cap():
            return self.update_all_layers_batch(sxA, sxV, svm,
                                                epoch_idx=epoch_idx, return_delayed=True)
        keys = _l6_complete_keys(self)
        S = _l6_snapshot(self, keys)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):                    # side-stream warmup (cuDNN/allocator)
            for _ in range(3):
                call_cap()
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
        _l6_restore(self, S); torch.cuda.synchronize()  # undo side-stream warmup mutations
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call_cap()
        torch.cuda.synchronize()
        return g, sxA, sxV, svm

    def _l6_install(self, B, epoch_idx):
        """Capture-last install of the L6 train graph for the CURRENT phase, then FULLY
        restore the pre-capture weights+state+RNG so real training is byte-identical to
        the eager path. Sets self._l6_forward (per-ext-step replay closure) and
        self._l6_phase=(epoch_idx>25, B). Called once per phase, right after the
        alloc-free reset_state(B), so the graph capture is the last allocation event
        before the replays (== the proven single-graph capture-last config)."""
        self._lever_L4_graph = True                    # L6 needs the L4 substrate (gpos+reset)
        # persistent static OUTPUT buffers the captured loop writes the final-substep spikes
        # into (in-place) — stable addresses shared by every replay (nothing escapes capture)
        for nm in ('_g_out_sA', '_g_out_sV', '_g_out_sMSI',
                   '_g_out_dA2M', '_g_out_dV2M', '_g_out_sOut'):
            t = getattr(self, nm, None)
            if t is None or t.shape[0] != B:
                setattr(self, nm, torch.zeros((B, self.n), device=self.device))
        # _g_out_sMSI_inh is the (B, n_inh) inh-population sibling of the (B, n) buffers
        # above (different width). The final-substep inh spikes (_latest_sMSI_inh = sMi)
        # ESCAPE the capture just like sA/sV/sMSI, so give them a persistent static-address
        # buffer too and repoint _latest_sMSI_inh at it post-replay (#66-proven completeness
        # bookkeeping: stale escaped pointer, NOT a dynamics change — the in-graph iSTDP that
        # consumes sMi is already byte-identical to eager).
        ti = getattr(self, '_g_out_sMSI_inh', None)
        if ti is None or ti.shape[0] != B:
            self._g_out_sMSI_inh = torch.zeros((B, self.n_inh),
                                               dtype=torch.float32, device=self.device)
        # snapshot the COMPLETE pre-capture state (post-reset init + current weights) + RNG
        keys = _l6_complete_keys(self)
        S0 = _l6_snapshot(self, keys)
        RNG0 = _l6_snapshot_rng()
        g, sxA, sxV, svm = self._l6_capture_phase(B, epoch_idx, self.g_rec)
        self._cap_stop_after_substeps = False          # restore eager-path flags
        self._cap_skip_dbg_sync = False
        _l6_restore(self, S0); torch.cuda.synchronize()  # in-place -> graph addresses preserved
        _l6_restore_rng(RNG0)

        B_cap = B

        def _forward(xA_b, xV_b, valid_mask=None, epoch_idx=epoch_idx,
                     return_delayed=True, **kw):
            # a non-captured shape (e.g. a B<batch_size tail mini-batch) -> eager fallback
            if xA_b.shape[0] != B_cap:
                return self.update_all_layers_batch(
                    xA_b, xV_b, valid_mask, epoch_idx=epoch_idx,
                    return_delayed=return_delayed, **kw)
            sxA.copy_(xA_b); sxV.copy_(xV_b)
            if valid_mask is not None:
                svm.copy_(valid_mask)
            g.replay()
            # The EAGER post-loop plasticity tail reads self._latest_*; after replay those
            # attrs hold ESCAPED graph-internal addresses (stale). Repoint them at the
            # PERSISTENT _g_out_* buffers the captured loop fills in-place every replay (the
            # proven panel-wrapper fix) so the tail reads this step's true final-substep spikes.
            self._latest_sA = self._g_out_sA;   self._latest_sV = self._g_out_sV
            self._latest_sMSI = self._g_out_sMSI; self._latest_sOut = self._g_out_sOut
            self._latest_sMSI_inh = self._g_out_sMSI_inh   # #66: repoint stale escaped inh ptr
            self._latest_dA2M = self._g_out_dA2M; self._latest_dV2M = self._g_out_dV2M
            # EAGER post-loop plasticity — identical call sequence to update_all_layers_batch
            # (the post-substep-loop block); the in-loop anchor already ran inside the graph.
            if getattr(self, 'plasticity_enabled', True):
                apply_topographic_anchor_unimodal(self, layer="A", lr=1.0 * self.lr_uni, sigma=2.5)
                apply_topographic_anchor_unimodal(self, layer="V", lr=1.0 * self.lr_uni, sigma=2.5)
                apply_local_competition_unimodal_fast(self, "A", beta=2.0 * self.lr_uni, neighbour_dist=4)
                apply_local_competition_unimodal_fast(self, "V", beta=2.0 * self.lr_uni, neighbour_dist=4)
                if epoch_idx > 25:
                    apply_local_competition_msi_fast(self, beta=1.5 * self.lr_msi, neighbour_dist=6)
                soft_row_scaling(self)
            return (self._g_out_sA, self._g_out_sV, self._g_out_sMSI,
                    self._g_out_sOut, self._g_out_dA2M, self._g_out_dV2M)

        self._l6_forward = _forward
        self._l6_phase = (epoch_idx > 25, B)

    def _l7_install(self, B, epoch_idx):
        """#63: capture-last install of the L7 STDP-tail graph for the CURRENT phase, then
        FULLY restore pre-capture weights+state+RNG so eager training stays byte-identical.
        Must run AFTER _l6_install THIS phase — the captured tail reads the address-stable
        _g_out_* spikes the forward graph fills in-place. The only NON-static inputs to the
        tail — the Poisson presyn spikes pre_inA/pre_inV (RNG, eager) and the ep>25 1-step
        recurrent pre — are staged into persistent buffers (_s_pre_inA/_s_pre_inV/_s_prev_rec)
        the captured graph reads at baked addresses; the per-ext-step wrapper copy_'s into them.
        Sets self._l7_stdp_graph + self._l7_phase=(epoch_idx>25, B). Mirrors the proven
        _l6 capture-last machinery; nothing escapes the capture (writes are in-place into
        persistent nn.Parameters + trace attrs), so it coexists with the L6 forward graph the
        same way run34.install_l4 captures its two phase graphs back-to-back."""
        self._lever_L6_graph_train = True              # L7 needs the L6 forward graph (_g_out_*)
        self._lever_L4_graph       = True              # ... which needs the L4 substrate
        # persistent STATIC input buffers the captured STDP tail reads (stable addresses):
        for nm in ('_s_pre_inA', '_s_pre_inV', '_s_prev_rec'):
            t = getattr(self, nm, None)
            if t is None or t.shape[0] != B:
                setattr(self, nm, torch.zeros((B, self.n), device=self.device))
        # the captured tail reads the _g_out_* spikes at the SAME addresses the L6 forward fills,
        # plus the static staging buffers — pass those persistent tensors (NOT fresh ones) so the
        # graph bakes the right addresses. ep>25 -> recurrent pre = _s_prev_rec (static).
        def _call_tail():
            self._stdp_tail(self._g_out_sA, self._g_out_sV, self._g_out_sMSI,
                            self._g_out_dA2M, self._g_out_dV2M,
                            self._s_pre_inA, self._s_pre_inV, self._s_prev_rec,
                            epoch_idx, debug=False)
        # snapshot COMPLETE pre-capture state + RNG (the warmup AND the capture-run both mutate
        # weights+traces); restoring undoes both -> the install is byte-transparent.
        keys = _l6_complete_keys(self)
        S0 = _l6_snapshot(self, keys)
        RNG0 = _l6_snapshot_rng()
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):                    # side-stream warmup (allocator/kernels)
            for _ in range(3):
                _call_tail()
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
        _l6_restore(self, S0); torch.cuda.synchronize()   # undo warmup before the capture
        g_stdp = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_stdp):                 # capture LAST (no alloc before replay)
            _call_tail()
        torch.cuda.synchronize()
        _l6_restore(self, S0); torch.cuda.synchronize()   # undo the capture-run mutation
        _l6_restore_rng(RNG0)
        self._l7_stdp_graph = g_stdp
        self._l7_phase = (epoch_idx > 25, B)

    # ─────────────────────────────────────────────────────────
    # #69 LEVER L8 — capture & replay the FUSED single graph (forward + plasticity + STDP tail)
    # ─────────────────────────────────────────────────────────
    def _l8_capture_phase(self, B, epoch_idx, g_rec):
        """#69/#74 LEVER L8 (PATH A, byte-identical partial fusion) — capture the single graph for
        one phase up to the post-loop ANCHOR: forward substeps + in-loop anchor + the two post-loop
        topographic anchors, captured TOGETHER (capture-last), STOPPING before the unimodal
        competition (_cap_stop_after_anchor). The competition's neighbour-mask matmul
        (s=matmul(M,r)) binds a different GEMM kernel under capture than eager → W_in diverges
        1.58e-4 (debugger #73, causal flip), so competition + msi-competition + soft_row_scaling +
        the STDP tail are EXCLUDED from the graph and run EAGER post-replay in _l8_forward (mirrors
        L6's certified discipline; once competition is eager everything after it must be too, else a
        2nd graph re-introduces #68). The in-graph _g_out_* write-back fills the persistent output
        buffers the eager tail/competition read via the _latest_*→_g_out_* repoint. RNG- and
        weight-DIRTY; the caller (_l8_install) wraps it in a complete snapshot/restore so the net is
        left byte-unchanged. Returns (graph, sxA, sxV, svm)."""
        self.g_rec = g_rec
        loc, mod, ok, lens = generate_event_loc_seq_batch(
            batch_size=B, space_size=self.space_size,
            offset_probability=0.6, temporal_jitter_max=2)
        T = max(lens)
        inten = 10 ** torch.empty(1).uniform_(math.log10(0.4), 0).item()
        xA, xV, valid = generate_av_batch_tensor(
            loc, mod, ok, n=self.n, space_size=self.space_size, sigma_in=self.sigma_in,
            noise_std=self.noise_std, loc_jitter_std=self.loc_jitter_std,
            stimulus_intensity=inten, device=self.device, max_len=T)

        full = getattr(self, '_lever_L8_full', False)  # #74 PATH B: competition matmul IN-GRAPH
        self.reset_state(B)
        self._cap_skip_dbg_sync = True
        self._cap_stop_after_substeps = False          # run the full substep loop in-graph
        # PATH A (partial, byte-identical): stop right after the post-loop anchor -> competition/
        # msi/soft_row/STDP-tail run EAGER post-replay. PATH B (full, the <=10 relax): DON'T stop
        # -> capture the WHOLE body incl the competition matmul + the STDP tail (only the Poisson
        # presyn stays eager).
        self._cap_stop_after_anchor = (not full)
        K = min(6, T - 1)
        for t in range(K):                             # warm the full body -> builds kernel caches
            self.update_all_layers_batch(xA[:, t], xV[:, t], valid[:, t],
                                         epoch_idx=epoch_idx, return_delayed=True)
        sxA = xA[:, K].contiguous().clone()
        sxV = xV[:, K].contiguous().clone()
        svm = valid[:, K].contiguous().clone()
        if full:
            # PATH B: the STDP tail is captured IN the graph -> seed its static input buffers (only
            # the addresses are baked; the per-ext-step wrapper copy_'s the real values each replay).
            self._s_pre_inA.copy_(self.sample_poisson_spikes_from_analog(sxA, max_rate=300., dt=0.01))
            self._s_pre_inV.copy_(self.sample_poisson_spikes_from_analog(sxV, max_rate=300., dt=0.01))
            self._s_prev_rec.zero_()

        def call_cap():
            if full:
                # PATH B (full fusion): capture the ENTIRE per-ext-step body — substeps + ALL
                # post-loop plasticity (incl the unimodal competition matmul, the #73 GEMM that
                # diverges ~1.58e-4 in-graph — the accepted relax) + the STDP tail — as ONE graph.
                sA, sV, sM, sO, dA2M, dV2M = self.update_all_layers_batch(
                    sxA, sxV, svm, epoch_idx=epoch_idx, return_delayed=True)
                self._stdp_tail(sA, sV, sM, dA2M, dV2M,
                                self._s_pre_inA, self._s_pre_inV, self._s_prev_rec,
                                epoch_idx, debug=False)
                return
            # #74 PATH A: capture substeps + in-loop anchor + POST-LOOP ANCHOR ONLY
            # (_cap_stop_after_anchor returns right after the two anchors). The in-graph _g_out_*
            # write-back fills the persistent output buffers; the unimodal competition,
            # msi-competition, soft_row_scaling and the STDP tail all run EAGER post-replay in
            # _l8_forward — the competition's neighbour matmul is the in-graph-vs-eager GEMM
            # divergence (#73), and once it is eager everything after it must be too (a 2nd graph
            # would re-introduce the #68 two-graph hazard).
            self.update_all_layers_batch(
                sxA, sxV, svm, epoch_idx=epoch_idx, return_delayed=True)

        keys = _l6_complete_keys(self)
        S = _l6_snapshot(self, keys)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):                    # side-stream warmup (cuDNN/allocator)
            for _ in range(3):
                call_cap()
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
        _l6_restore(self, S); torch.cuda.synchronize()  # undo side-stream warmup mutations
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):                      # capture LAST (no alloc before replay)
            call_cap()
        torch.cuda.synchronize()
        return g, sxA, sxV, svm

    def _l8_install(self, B, epoch_idx):
        """#69: capture-last install of the FUSED single graph (forward + post-loop plasticity +
        STDP tail) for the CURRENT phase, then FULLY restore the pre-capture weights+state+RNG so
        real training is byte-identical to the L6+eager-tail path. ONE graph only -> avoids the
        #68 two-graph use-after-free entirely (no separate g_stdp). Sets self._l8_forward (the
        per-ext-step replay closure) + self._l8_phase=(epoch_idx>25, B). Allocates BOTH the L6
        _g_out_* output buffers (the in-graph write-back fills them; the recurrent advance reads
        _g_out_sMSI) AND the L7 static tail-input buffers (_s_pre_inA/_s_pre_inV/_s_prev_rec)."""
        self._lever_L4_graph = True                    # L8 needs the L4 substrate (gpos + reset)
        # persistent OUTPUT buffers the in-graph _g_out write-back fills in-place (stable addresses)
        for nm in ('_g_out_sA', '_g_out_sV', '_g_out_sMSI',
                   '_g_out_dA2M', '_g_out_dV2M', '_g_out_sOut'):
            t = getattr(self, nm, None)
            if t is None or t.shape[0] != B:
                setattr(self, nm, torch.zeros((B, self.n), device=self.device))
        ti = getattr(self, '_g_out_sMSI_inh', None)
        if ti is None or ti.shape[0] != B:
            self._g_out_sMSI_inh = torch.zeros((B, self.n_inh),
                                               dtype=torch.float32, device=self.device)
        # persistent STATIC tail-input buffers (Poisson presyn + ep>25 1-step recurrent pre)
        for nm in ('_s_pre_inA', '_s_pre_inV', '_s_prev_rec'):
            t = getattr(self, nm, None)
            if t is None or t.shape[0] != B:
                setattr(self, nm, torch.zeros((B, self.n), device=self.device))
        # snapshot the COMPLETE pre-capture state (post-reset init + weights) + RNG; the warmup AND
        # the capture-run both mutate weights/traces/RNG -> restoring undoes both (byte-transparent).
        keys = _l6_complete_keys(self)
        S0 = _l6_snapshot(self, keys)
        RNG0 = _l6_snapshot_rng()
        g, sxA, sxV, svm = self._l8_capture_phase(B, epoch_idx, self.g_rec)
        self._cap_stop_after_substeps = False          # restore eager-path flags
        self._cap_stop_after_anchor = False            # #74: clear the PATH A capture hook
        self._cap_skip_dbg_sync = False
        _l6_restore(self, S0); torch.cuda.synchronize()  # in-place -> graph addresses preserved
        _l6_restore_rng(RNG0)

        B_cap = B
        full = getattr(self, '_lever_L8_full', False)   # #74 PATH B: full fusion (competition in-graph)

        def _forward(xA_b, xV_b, valid_mask, pre_inA, pre_inV, epoch_idx=epoch_idx):
            # _use_l8 guarantees B == batch_size == B_cap; the guard is defensive (a non-captured
            # shape signals the caller to fall back to eager — never happens on the bs=250 path).
            if xA_b.shape[0] != B_cap:
                return False
            sxA.copy_(xA_b); sxV.copy_(xV_b)
            if valid_mask is not None:
                svm.copy_(valid_mask)
            if full:
                # PATH B (full fusion): the WHOLE body — substeps + post-loop plasticity (incl the
                # in-graph competition matmul) + STDP tail — is captured; stage the Poisson presyn
                # into the captured tail's static buffers, then replay. Nothing eager remains.
                self._s_pre_inA.copy_(pre_inA); self._s_pre_inV.copy_(pre_inV)
                g.replay()
                if epoch_idx > 25:
                    self._s_prev_rec.copy_(self._g_out_sMSI)   # advance the 1-step recurrent pre
                return True
            g.replay()                                 # substeps + in-loop + post-loop ANCHOR (in-graph)
            # The captured graph stopped after the post-loop anchor; the escaped self._latest_*
            # now hold stale graph-internal addresses. Repoint them at the PERSISTENT _g_out_*
            # the captured loop fills in-place every replay (clears the benign #66 |Δ|=1.0 AND
            # feeds the eager competition/tail this step's true final-substep spikes — both read
            # _latest_*). Mirrors L6 (_l6_forward L3816-3819).
            self._latest_sA = self._g_out_sA;     self._latest_sV = self._g_out_sV
            self._latest_sMSI = self._g_out_sMSI; self._latest_sOut = self._g_out_sOut
            self._latest_sMSI_inh = self._g_out_sMSI_inh
            self._latest_dA2M = self._g_out_dA2M; self._latest_dV2M = self._g_out_dV2M
            # EAGER post-anchor plasticity — the unimodal competition's neighbour matmul is the
            # in-graph-vs-eager GEMM divergence (#73); run it + everything after it (msi-comp,
            # soft_row, STDP tail) EAGER, exactly mirroring L6's certified post-loop sequence.
            if getattr(self, 'plasticity_enabled', True):
                apply_local_competition_unimodal_fast(self, "A", beta=2.0 * self.lr_uni, neighbour_dist=4)
                apply_local_competition_unimodal_fast(self, "V", beta=2.0 * self.lr_uni, neighbour_dist=4)
                if epoch_idx > 25:
                    apply_local_competition_msi_fast(self, beta=1.5 * self.lr_msi, neighbour_dist=6)
                soft_row_scaling(self)
            # EAGER STDP tail (the per-ext-step ger sequence). pre_spk_rec = _s_prev_rec holds last
            # step's sMSI (zeroed at step 0 by the caller); it is read ONLY at ep>25 inside the tail
            # — pass None at ep<=25 to match the eager reference byte-for-byte.
            pre_spk_rec = self._s_prev_rec if epoch_idx > 25 else None
            self._stdp_tail(self._g_out_sA, self._g_out_sV, self._g_out_sMSI,
                            self._g_out_dA2M, self._g_out_dV2M,
                            pre_inA, pre_inV, pre_spk_rec, epoch_idx, debug=False)
            if epoch_idx > 25:
                # advance the 1-step recurrent pre for the NEXT step (this step's sMSI -> next pre).
                self._s_prev_rec.copy_(self._g_out_sMSI)
            return True

        self._l8_forward = _forward
        self._l8_phase = (epoch_idx > 25, B)

    # ─────────────────────────────────────────────────────────
    # #63 — the per-ext-step STDP weight-update tail, extracted VERBATIM from the
    # train_unsupervised_batch inner loop (the 6 FF stdp_update_batch ger sequences +
    # the ep>25 recurrent MSI->MSI ger + the AMPA/NMDA renorm). Single source of truth:
    # the eager path calls this directly (byte-identical to the old inline block — proven
    # by the port-fidelity/determinism/L4/L5/L6 gate checks), and lever L7 captures this
    # exact call once and replays it (no CPU dispatch of the ~1024 tiny ger kernels).
    # pre_spk_rec is passed IN (eager: self._prev_sMSI_rec/zeros; L7: the static _s_prev_rec
    # staging buffer); the cross-ext-step _prev_sMSI_rec update stays in the caller — it is
    # the only inter-step dependency, so it lives outside this single-step (captured) region.
    # ─────────────────────────────────────────────────────────
    def _stdp_tail(self, sA, sV, sMSI, dA2M, dV2M, pre_inA, pre_inV,
                   pre_spk_rec, epoch_idx, debug=False):
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
            self.pre_trace_msi_rec, self.post_trace_msi_rec, _ = self.stdp_update_batch(
                'W_MSI_exc',
                post_spk=sMSI,
                pre_spk=pre_spk_rec,
                post_trace=self.post_trace_msi_rec,
                pre_trace=self.pre_trace_msi_rec,
                lr=self.lr_msi,
                debug=debug)
            self.W_MSI_exc.data.fill_diagonal_(0.0)

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

        _l8 = getattr(self, '_lever_L8_fused', False)         # #69: fused forward+plasticity+STDP single graph
        if _l8:
            self._lever_L4_graph = True                       # L8 needs the L4 graphing substrate
            _l6 = _l7 = False                                 # L8 is mutually exclusive with L6/L7 (never 2 graphs, #68)
        else:
            _l6 = getattr(self, '_lever_L6_graph_train', False)   # #62: graph the substep loop
            _l7 = getattr(self, '_lever_L7_graph_stdp', False)    # #63: graph the per-step STDP tail
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

            # #69 LEVER L8: capture-last install of the FUSED single graph (forward substeps +
            # post-loop plasticity + STDP tail), right AFTER the alloc-free reset so the capture is
            # the last allocation event before the replays. ONE graph only -> no #68 two-graph hazard.
            if _l8 and B == batch_size and getattr(self, '_l8_phase', None) != (epoch_idx > 25, B):
                self._l8_install(B, epoch_idx)
            _use_l8 = _l8 and getattr(self, '_l8_phase', None) == (epoch_idx > 25, B)
            if _use_l8 and epoch_idx > 25:
                self._s_prev_rec.zero_()   # step-0 recurrent pre = zeros (mirrors eager _prev=None)

            # LEVER L6 (#62): capture-last install of the train substep graph on the first
            # full-batch mini-batch of this phase — right AFTER the alloc-free reset, so the
            # CUDA-graph capture is the last allocation event before the replays. State + RNG
            # are snapshotted/restored across the capture, so this is byte-identical to eager.
            if _l6 and B == batch_size and getattr(self, '_l6_phase', None) != (epoch_idx > 25, B):
                self._l6_install(B, epoch_idx)
            _use_l6 = _l6 and getattr(self, '_l6_phase', None) == (epoch_idx > 25, B)
            _fwd = self._l6_forward if _use_l6 else self.update_all_layers_batch

            # #63 LEVER L7: capture-last install of the STDP-tail graph, AFTER the L6 forward
            # graph is live this phase (the captured tail reads the address-stable _g_out_*
            # the forward fills). Panel-measurement (the dW flux accumulator) forces the eager
            # tail — its .item() host sync can't be captured — so L7 only engages when
            # _panel_dW_accum is None (always true on the plain training path).
            _use_l7 = (_l7 and _use_l6 and B == batch_size
                       and getattr(self, '_panel_dW_accum', None) is None)
            if _use_l7 and getattr(self, '_l7_phase', None) != (epoch_idx > 25, B):
                self._l7_install(B, epoch_idx)
            _use_l7 = _use_l7 and getattr(self, '_l7_phase', None) == (epoch_idx > 25, B)
            if _use_l7 and epoch_idx > 25:
                self._s_prev_rec.zero_()   # step-0 recurrent pre = zeros (mirrors eager _prev=None)

            for t in range(T_max):
                if _use_l8:
                    # #69 FUSED single graph: sample the Poisson presyn spikes EAGERLY (the only
                    # per-step RNG) BEFORE the atomic replay and stage them — the captured forward is
                    # RNG-free, so Poisson-before-forward is byte-identical. The replay runs the WHOLE
                    # body (forward substeps + post-loop plasticity + STDP tail) in one graph; the
                    # ep>25 1-step recurrent pre is advanced inside _l8_forward. Nothing eager remains.
                    pre_inA = self.sample_poisson_spikes_from_analog(xA[:, t], max_rate=300., dt=0.01)
                    pre_inV = self.sample_poisson_spikes_from_analog(xV[:, t], max_rate=300., dt=0.01)
                    self._l8_forward(xA[:, t], xV[:, t], valid[:, t], pre_inA, pre_inV,
                                     epoch_idx=epoch_idx)
                    continue
                # forward pass  (L6 ON -> replay the captured substep graph; OFF -> eager)
                (sA, sV, sMSI, _,
                 dA2M, dV2M) = _fwd(
                    xA[:, t],  # analog A input
                    xV[:, t],  # analog V input
                    valid[:, t],  # validity mask
                    epoch_idx=epoch_idx,
                    return_delayed=True)

                pre_inA = self.sample_poisson_spikes_from_analog(
                    xA[:, t], max_rate=300., dt=0.01)
                pre_inV = self.sample_poisson_spikes_from_analog(
                    xV[:, t], max_rate=300., dt=0.01)

                if _use_l7:
                    # #63: GRAPHED STDP tail — stage the RNG presyn spikes into the static
                    # buffers and replay the captured ger sequence (zero CPU dispatch). The
                    # forward (L6 replay) just refreshed _g_out_* with this step's spikes that
                    # the captured tail reads; _s_prev_rec already holds last step's sMSI as the
                    # ep>25 1-step recurrent pre.
                    self._s_pre_inA.copy_(pre_inA)
                    self._s_pre_inV.copy_(pre_inV)
                    self._l7_stdp_graph.replay()
                    if epoch_idx > 25:
                        # advance the 1-step recurrent pre for the NEXT step (eager; the only
                        # cross-ext-step dependency, kept outside the captured single-step graph).
                        self._s_prev_rec.copy_(self._g_out_sMSI)
                else:
                    # #63: ep>25 recurrent pre-spike (1-step conduction delay). Read BEFORE the
                    # extracted tail; the _prev_sMSI_rec update stays AFTER it. That 1-step
                    # cross-ext-step dependency is the only state the captured single-step L7
                    # graph can't own, so it lives in the loop (eager) on both paths.
                    pre_spk_rec = None
                    if epoch_idx > 25:
                        pre_spk_rec = (self._prev_sMSI_rec if self._prev_sMSI_rec is not None
                                       else torch.zeros_like(sMSI))  # REVERT(debugger#60): delayed pre

                    # per-ext-step STDP weight-update tail (extracted verbatim; L7 captures
                    # this exact call once and replays it — see _stdp_tail / _l7_install).
                    self._stdp_tail(sA, sV, sMSI, dA2M, dV2M, pre_inA, pre_inV,
                                    pre_spk_rec, epoch_idx, debug)

                    if epoch_idx > 25:
                        # L6 (#62): under the graph, sMSI ALIASES the persistent _g_out_sMSI buffer
                        # (overwritten by the NEXT replay), so clone to freeze THIS step's spikes for
                        # the 1-step recurrent delay. The eager path gets a fresh per-step tensor, so
                        # .clone() is byte-identical there -> gate on L6 to keep eager alloc-free.
                        self._prev_sMSI_rec = (sMSI.detach().clone() if _use_l6
                                               else sMSI.detach())  # REVERT(debugger#60): delayed pre

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

        # task #192 Phase E: D'Amour-Froemke symmetric iSTDP on the
        # MSI_inh -> MSI_exc GABA edge (W_msiInh2Exc_GABA). See spec
        # sci_inhibition_functional_role.md §5.4b.
        tau_istdp_pre=net.tau_istdp_pre,
        tau_istdp_post=net.tau_istdp_post,
        eta_istdp=net.eta_istdp,
        istdp_baseline=net.istdp_baseline,
        W_gaba_clamp=net.W_gaba_clamp,

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
        net.gNMDA = 1.30
        net.tau_nmda = 80.0
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

        if panel and not net._lever_L1_drop_inloop_log:
            # task#21/E6b: Tap-A taps — FF dW-flux accumulator + in-training E/I
            # recording around train_unsupervised_batch (both read-only).
            # LEVER L1: these recorders drive per-substep .item() host-syncs in the
            # forward/STDP hot loop (3.0x slowdown). Skipping them here leaves
            # _panel_dW_accum / _ei_record as None, so the in-loop logging blocks are
            # strict no-ops; the every-5 panel battery (_epoch_panel_dump) still runs.
            net._panel_dW_accum = {nm: 0.0 for nm in
                ('W_inA', 'W_inV', 'W_a2msi_AMPA', 'W_v2msi_AMPA',
                 'W_a2msi_NMDA', 'W_v2msi_NMDA')}
            net.start_ei_recording()

        # Increment 3: recurrent MSI->MSI excitation schedule — OFF until the
        # post-ep25 MSI plasticity stage, then static g_rec=0.1, matching the
        # epoch_idx>25 gate on the recurrent STDP (0-indexed: ON from epoch 26).
        net.g_rec = 0.1 if epoch > 25 else 0.0
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

