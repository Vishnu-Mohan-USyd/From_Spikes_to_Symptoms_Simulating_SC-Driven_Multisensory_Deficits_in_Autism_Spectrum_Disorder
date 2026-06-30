#!/usr/bin/env python3
"""PANEL CAPTURE-HARNESS (#49 build phase). A separate panel-net built from
Training_graphdf (delay-fix forward + L1/L2/L4 graph + capture-safe recording),
into which we load the TRAINING net's weights each measurement epoch; its panel
forward runs as TWO CUDA graphs (B=8 single-volley w/ recording, B=540 TBW w/o)
so the per-epoch panel stops being a ~55 s launch-bound sequential tail.

This module exposes:
  build_net(MOD, seed, batch_size)        — verbatim panel build (matches AS-IS baseline)
  transfer_weights(src, dst)              — in-place copy of all batch-independent
                                            float tensors (== production state_dict load)
  setup_panel_net(net, MOD, g_rec)        — alloc rec buffers, capture both graphs, install wrapper
  run_panel(net, seed)                    — seed all 3 RNGs, run _panel_battery, return (out, secs)

All compute cuda:0 / RTX 5090. Throwaway — no production build is committed here.
"""
import os, sys, time, math, copy, random
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))  # flat
CODE = ROOT
BASE = ROOT   # AS-IS Training_delayfix.py
HERE = ROOT
LEV  = ROOT
for p in (CODE, BASE, HERE, LEV):
    if p not in sys.path:
        sys.path.insert(0, p)

import run34 as R   # reuse snapshot/restore/complete_keys/_resolve/snapshot_rng helpers

EI_KEYS  = ["I_E_mean", "I_I_mean", "Q_E", "Q_I", "AMPA", "NMDA", "FFInh", "RecurInh", "LatInh"]
INH_KEYS = ["mg_iA", "mg_iV", "mg_iA_on", "mg_iV_on", "v_dend_inhA", "v_dend_inhV"]
ALL_REC  = EI_KEYS + INH_KEYS

VITALS = [("v_rate", "P5_active_mean_hz"), ("v_EI", "P2B_EI_ratio"),
          ("v_F0", "P1_P_at_0"), ("v_TBW", "P1_width_ms")]


# ───────────────────────── build (verbatim panel build) ──────────────────────
def build_net(MOD, seed, batch_size=540):
    torch.manual_seed(int(seed)); np.random.seed(int(seed)); random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    net = MOD.MultiBatchAudVisMSINetworkTime(
        n_neurons=180, batch_size=batch_size,
        lr_unimodal=2e-2, lr_msi=2e-2, lr_readout=8e-4,
        sigma_in=10.0, sigma_teacher=2.0, noise_std=0.02, single_modality_prob=0.5,
        v_thresh=0.3, dt=0.1, tau_m=20.0, n_substeps=100, loc_jitter_std=0,
        space_size=180, conduction_delay_a2msi=250, conduction_delay_v2msi=400)
    net.set_inhib_plasticity(True)
    MOD.assign_unimodal_preferred_locations(net)
    net.b_uniA.data.fill_(0.0); net.b_uniV.data.fill_(0.0)
    MOD._init(net.W_a2msi_AMPA, 0.004); MOD._init(net.W_v2msi_AMPA, 0.004)
    MOD._init(net.W_a2msi_NMDA, 0.004); MOD._init(net.W_v2msi_NMDA, 0.004)
    with torch.no_grad():
        net.gNMDA = float(os.environ.get("GNMDA", 1.30)); net.tau_nmda = float(os.environ.get("TAU_NMDA", "80.0")); net.nmda_alpha = 0.1   # #66 gNMDA + NR2A TAU_NMDA env-exposed (defaults 1.30/80.0 == byte-identical)
        net.tau_nmda_v = float(os.environ.get("TAU_NMDA_V", net.tau_nmda))   # latency #153: V-specific FF-NMDA decay (build_net/retrain path; default == net.tau_nmda => byte-identical; > tau_nmda => longer V sub-threshold window)
        net.Erev_nmda = 20.0; net.tau_nmdaVolt = 100.0; net.v_nmda_rest = -65.0
        net.nmda_vrest_offset = 7.0; net.mg_vhalf = float(os.environ.get("MG_VHALF", -35.0)); net.tau_nmda_inh = 21.6
        net.dend_coupling_alpha = float(os.environ.get("DEND_COUPLING_ALPHA", 0.1))   # P5 NMDA Mg-gate knob (env-config; default == original 0.1)
        net.mg_k = float(os.environ.get("MG_K", 0.062))                               # C1 NMDA Mg-gate slope (env-config; default == original 0.062)
        net.mg_vhalf_inh = float(os.environ.get("MG_VHALF_INH", net.mg_vhalf))         # inh-gate DECOUPLE (env-config; default == exc mg_vhalf => byte-identical)
    net.u_a.fill_(net.u_stp_a); net.u_v.fill_(net.u_stp_v); net.tau_rec = float(os.environ.get("TAU_REC", 400.0))  # task#104: dead 0.7 fill -> env-exposed operative U (reset_state re-applies); tau_rec env-exposed (defaults 0.2/400 byte-identical)
    net.input_scaling = 400; net.g_GABA = float(os.environ.get("G_GABA", "10"))   # #94/#92 inhibition lever (env-config; default == original 10 => byte-identical)
    return net


def transfer_weights(src, dst):
    """In-place copy of EVERY batch-independent float tensor (attrs+params+buffers)
    from src into dst by name (== loading the training net's state into the panel-net).
    Batch-dependent transient state (shape mismatch) is skipped — the panel resets it."""
    n_copy = n_skip = 0
    for kind, name in R.complete_keys(src):
        s = R._resolve(src, kind, name)
        try:
            d = R._resolve(dst, kind, name)
        except Exception:
            n_skip += 1; continue
        if torch.is_tensor(s) and torch.is_tensor(d) and s.shape == d.shape:
            d.copy_(s); n_copy += 1
        else:
            n_skip += 1
    return n_copy, n_skip


def sd_maxdiff(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    worst = 0.0; worstk = None
    for k in sa:
        if k in sb and sa[k].shape == sb[k].shape:
            d = (sa[k].double() - sb[k].double()).abs().max().item()
            if d > worst:
                worst, worstk = d, k
    return worst, worstk


# ───────────────────────── graph capture (panel shapes) ──────────────────────
GOUT_NAMES = ('sA', 'sV', 'sMSI', 'sOut', 'dA2M', 'dV2M', 'sumM')

def _alloc_gouts(net, B):
    return {nm: torch.zeros((B, net.n), device=net.device) for nm in GOUT_NAMES}

def _bind_gouts(net, g):
    net._g_out_sA = g['sA']; net._g_out_sV = g['sV']; net._g_out_sMSI = g['sMSI']
    net._g_out_sOut = g['sOut']; net._g_out_dA2M = g['dA2M']; net._g_out_dV2M = g['dV2M']
    net._g_out_sumM = g['sumM']


def capture_panel_graph(net, MOD, B, g_rec, record, pool=None):
    """Capture the loop-only substep graph at panel shape B, with return_spike_sum=True
    (TBW needs ret[-1]=sum_sM) and the recording branch baked iff `record`.
    `pool` (a graph_pool_handle) shares one reserved memory arena across BOTH panel
    graphs so the second capture cannot stomp the first graph's baked addresses —
    safe because TBW and volley are replayed sequentially, never interleaved."""
    net.g_rec = g_rec
    net.plasticity_enabled = False
    net.enable_probe = False
    net._ei_record = {k: [] for k in EI_KEYS} if record else None
    net._panel_inh_accum = {k: [] for k in INH_KEYS} if record else None

    loc, mod, ok, lens = MOD.generate_event_loc_seq_batch(
        batch_size=B, space_size=net.space_size,
        offset_probability=0.6, temporal_jitter_max=2)
    T = max(lens)
    inten = 10 ** torch.empty(1).uniform_(math.log10(0.4), 0).item()
    xA, xV, valid = MOD.generate_av_batch_tensor(
        loc, mod, ok, n=net.n, space_size=net.space_size, sigma_in=net.sigma_in,
        noise_std=net.noise_std, loc_jitter_std=net.loc_jitter_std,
        stimulus_intensity=inten, device=net.device, max_len=T)

    net.reset_state(B)
    net._cap_skip_dbg_sync = True
    K = min(6, T - 1)
    net._cap_stop_after_substeps = False
    for t in range(K):
        net.update_all_layers_batch(xA[:, t], xV[:, t], valid[:, t],
                                    epoch_idx=0, return_spike_sum=True)
    sxA = xA[:, K].contiguous().clone()
    sxV = xV[:, K].contiguous().clone()
    svm = valid[:, K].contiguous().clone()

    net._cap_stop_after_substeps = True
    def call_cap():
        return net.update_all_layers_batch(sxA, sxV, svm, epoch_idx=0, return_spike_sum=True)

    keys = R.complete_keys(net)
    S = R.snapshot(net, keys)
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            call_cap()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    R.restore(net, S); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    if pool is not None:
        with torch.cuda.graph(g, pool=pool):
            call_cap()
    else:
        with torch.cuda.graph(g):
            call_cap()
    torch.cuda.synchronize()
    net._cap_stop_after_substeps = False
    return g, sxA, sxV, svm


def make_panel_wrapper(net, ORIG, vol, vol_g, tbw, tbw_g):
    vg, vxA, vxV, vvm = vol
    tg, txA, txV, tvm = tbw
    def wrapper(xA_b, xV_b, valid_mask=None, epoch_idx=0,
                return_delayed=False, return_spike_sum=False, **kw):
        B = xA_b.shape[0]
        if B == 8:
            vxA.copy_(xA_b); vxV.copy_(xV_b)
            if valid_mask is not None: vvm.copy_(valid_mask)
            vg.replay()
            if net._ei_record is not None:
                rg = net._rec_gpu
                for k in EI_KEYS:
                    net._ei_record[k].extend(rg[k].double().tolist())
            if net._panel_inh_accum is not None:
                rg = net._rec_gpu
                for k in INH_KEYS:
                    net._panel_inh_accum[k].extend(rg[k].double().tolist())
            return (vol_g['sA'], vol_g['sV'], vol_g['sMSI'], vol_g['sOut'], vol_g['sumM'])
        if B == 540:
            txA.copy_(xA_b); txV.copy_(xV_b)
            if valid_mask is not None: tvm.copy_(valid_mask)
            tg.replay()
            return (tbw_g['sA'], tbw_g['sV'], tbw_g['sMSI'], tbw_g['sOut'], tbw_g['sumM'])
        return ORIG(xA_b, xV_b, valid_mask, epoch_idx=epoch_idx,
                    return_delayed=return_delayed, return_spike_sum=return_spike_sum, **kw)
    return wrapper


VOLLEY_FRAMES = 120   # _panel_single_volley(B=8, T_frames=120) -> xA.shape[1]


def setup_panel_net(net, MOD, g_rec):
    """LAZY RE-CAPTURE volley graph (Lead decision after debugger #55).

    The setup-time capture is corrupted because the eager TBW (B=540) forward runs
    BETWEEN the volley capture and its first replay every panel (proven use-after-free,
    #55). Fix: capture the volley graph INSIDE the panel, AFTER the eager TBW completes,
    right before the 120-frame volley loop — so the capture is the LAST allocation event
    before the replays (== diag_single's proven-clean single-graph config). Re-captured
    every panel (capture ~3s), since each panel's TBW re-corrupts the prior graph.

    Three mandatory correctness constraints (Lead):
      (1) CAPTURE-LAST: pre-build both reset caches here; re-capture, then the only thing
          before the replays is reset_state(8) — whose cache path is alloc-free (copy_ +
          in-place _sync_graph_pos.fill_), verified — so nothing allocates capture→replay.
      (2) RNG-NEUTRAL: capture_panel_graph advances torch/cuda/numpy RNG (its own stimuli);
          save all 3 states before the re-capture and restore after, so the volley sees the
          exact RNG it would without it (bit-identity).
      (3) ALLOC-FREE harvest: pre-allocate _rec_all[k]=zeros(F,n_substeps) ONCE; per frame
          copy_ _rec_gpu[k] into _rec_all[k][f] (no alloc between replays); after the loop
          one readout into _ei_record/_panel_inh_accum (frame-major, substep-minor)."""
    net._lever_L4_graph = True
    net.plasticity_enabled = False
    net.enable_probe = False
    # persistent per-substep recording buffers (batch-independent scalar reductions)
    net._rec_gpu = {k: torch.zeros(net.n_substeps, device=net.device) for k in ALL_REC}
    # (3) persistent per-frame harvest buffer — single alloc, reused every panel
    net._rec_all = {k: torch.zeros(VOLLEY_FRAMES, net.n_substeps, device=net.device)
                    for k in ALL_REC}
    # (1) PRE-BUILD both reset caches so every reset_state(540)/(8) is an in-place restore
    net.reset_state(540)
    net.reset_state(8)
    # persistent volley graph-output buffers (bound before each re-capture; no realloc)
    net._vol_gout = _alloc_gouts(net, 8)

    ORIG = net.update_all_layers_batch          # raw forward (used for warmup + capture)
    ORIG_STOP = net.stop_ei_recording
    st = {'g': None, 'in_volley': False, 'f': 0}

    def wrapper(xA_b, xV_b, valid_mask=None, epoch_idx=0,
                return_delayed=False, return_spike_sum=False, **kw):
        clk = getattr(net, '_panel_clk', None)   # gated CPU-wall split timing (inert; off=None)
        if xA_b.shape[0] == 8:
            if not st['in_volley']:
                # ---- first volley frame of THIS panel: re-capture (capture-last) ----
                _tr = time.time()
                rng_cpu = torch.get_rng_state()
                rng_cuda = torch.cuda.get_rng_state_all()
                rng_np = np.random.get_state()                       # (2) save RNG
                _bind_gouts(net, net._vol_gout)                      # graph writes persistent g-outs
                net.update_all_layers_batch = ORIG                   # warmup/capture use RAW forward
                try:
                    st['g'] = capture_panel_graph(net, MOD, B=8, g_rec=net.g_rec, record=True)
                finally:
                    net.update_all_layers_batch = wrapper            # restore replay wrapper
                torch.set_rng_state(rng_cpu)
                torch.cuda.set_rng_state_all(rng_cuda)
                np.random.set_state(rng_np)                          # (2) restore RNG
                net.reset_state(8)                                   # frame-0 init (alloc-free cache path)
                st['in_volley'] = True
                st['f'] = 0
                if clk is not None:
                    clk['recap'] += time.time() - _tr
            _tv = time.time()
            vg, vxA, vxV, vvm = st['g']
            vxA.copy_(xA_b); vxV.copy_(xV_b)
            if valid_mask is not None:
                vvm.copy_(valid_mask)
            vg.replay()
            f = st['f']
            for k in ALL_REC:                                        # (3) alloc-free per-frame copy
                net._rec_all[k][f].copy_(net._rec_gpu[k])
            st['f'] = f + 1
            if clk is not None:
                clk['vol'] += time.time() - _tv; clk['nvol'] += 1
            go = net._vol_gout
            return (go['sA'], go['sV'], go['sMSI'], go['sOut'], go['sumM'])
        # TBW (B=540) / anything else -> eager; mark volley phase ended so the next
        # B=8 (next panel) re-captures AFTER this panel's eager TBW.
        st['in_volley'] = False
        _tt = time.time()
        r = ORIG(xA_b, xV_b, valid_mask, epoch_idx=epoch_idx,
                 return_delayed=return_delayed, return_spike_sum=return_spike_sum, **kw)
        if clk is not None:
            clk['tbw'] += time.time() - _tt; clk['ntbw'] += 1
        return r

    def stop_harvest():
        # (3) one readout after all replays: _rec_all -> recording dicts (frame-major,
        # substep-minor == baseline append order; float32->float64 widening is exact).
        f = st['f']
        if f > 0:
            if net._ei_record is not None:
                for k in EI_KEYS:
                    net._ei_record[k] = net._rec_all[k][:f].double().reshape(-1).tolist()
            if net._panel_inh_accum is not None:
                for k in INH_KEYS:
                    net._panel_inh_accum[k] = net._rec_all[k][:f].double().reshape(-1).tolist()
        st['in_volley'] = False
        return ORIG_STOP()

    net.update_all_layers_batch = wrapper
    net.stop_ei_recording = stop_harvest
    return net


# ───────────────────────── run + compare ─────────────────────────────────────
def run_panel(net, seed, g_rec):
    net.g_rec = g_rec
    net.plasticity_enabled = False; net.enable_probe = False
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.synchronize()
    t0 = time.time()
    out = net._panel_battery(seed, full=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.time() - t0
