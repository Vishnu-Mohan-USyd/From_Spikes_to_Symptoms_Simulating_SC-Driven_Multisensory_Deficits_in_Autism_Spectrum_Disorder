#!/usr/bin/env python3
"""#74 SPEED LADDER — measured per-epoch TRAIN wall for 3 variants at BOTH g_rec regimes (bs=250),
projected to an 80-epoch wall = 26*(g_rec=0 per-ep) + 54*(g_rec=0.1 per-ep), vs the 600s kill.

Variants (all bs=250 -> 4 even mini-batches of 250, NO eager partial):
  L6          : un-fused certified build (substep loop graphed; post-loop plasticity + STDP EAGER)
  L8-partial  : #74 PATH A byte-identical fix (substeps + post-loop anchor graphed; competition +
                soft_row + STDP tail EAGER) — gate_fused.py byte-PASSes this vs L6/delayfix
  L8-full     : competition IN-GRAPH (full fusion) — BYTE-FAILS (#73), so TIMING-ONLY via a
                MethodType monkeypatch that restores the original full-fusion _l8_install in memory
                (the in-file PATH-A methods are NOT touched; this only rebinds the throwaway net).

PRE-REGISTERED KILL (fixed before measuring): a variant's projected 80-ep wall >= 600s -> dead for
<=10min. Per-ep = steady-state mean (capture epoch + 1 settling epoch discarded -> last 3 of 5).
The g_rec=0.1 regime is timed on a fresh net run directly at ep26..30: per-epoch WORK (kernels /
replays / eager launches) is weight-independent, so the wall equals a real ep26+ epoch (the gate
already proved a fresh net trains correctly at ep26). cuda:0 only; delayfix frozen 5e7d6d20.
"""
import os, sys, time, json, math, random
import numpy as np
import torch
import importlib
from types import MethodType
import faulthandler; faulthandler.enable()
import panelgraph as PG

GDF = importlib.import_module("Training_graphdf")

SEED, N_SEQ, BS = 42, 1000, 250
GREC0_EPS  = [0, 1, 2, 3, 4]       # g_rec=0   (le25); capture @ ep0
GREC01_EPS = [26, 27, 28, 29, 30]  # g_rec=0.1 (gt25); capture @ ep26
N_DISCARD = 2                      # drop capture epoch + 1 settling -> steady = last 3


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ─────────── L8-FULL timing reconstruction (original full fusion; competition IN-GRAPH) ───────────
# Verbatim restore of the pre-#74 _l8_capture_phase/_l8_install (md5 c4fea86): _cap_stop_after_anchor
# stays False so the WHOLE update_all_layers_batch (incl the unimodal competition matmul) + the STDP
# tail are captured into ONE graph; _l8_forward only replays + advances the recurrent pre (NO eager
# plasticity). This is the byte-FAILing full-fusion build — used here purely to measure its wall.
def _l8_capture_full(self, B, epoch_idx, g_rec):
    self.g_rec = g_rec
    loc, mod, ok, lens = GDF.generate_event_loc_seq_batch(
        batch_size=B, space_size=self.space_size,
        offset_probability=0.6, temporal_jitter_max=2)
    T = max(lens)
    inten = 10 ** torch.empty(1).uniform_(math.log10(0.4), 0).item()
    xA, xV, valid = GDF.generate_av_batch_tensor(
        loc, mod, ok, n=self.n, space_size=self.space_size, sigma_in=self.sigma_in,
        noise_std=self.noise_std, loc_jitter_std=self.loc_jitter_std,
        stimulus_intensity=inten, device=self.device, max_len=T)
    self.reset_state(B)
    self._cap_skip_dbg_sync = True
    self._cap_stop_after_substeps = False
    self._cap_stop_after_anchor = False                # L8-FULL: competition stays IN-GRAPH
    K = min(6, T - 1)
    for t in range(K):
        self.update_all_layers_batch(xA[:, t], xV[:, t], valid[:, t],
                                     epoch_idx=epoch_idx, return_delayed=True)
    sxA = xA[:, K].contiguous().clone()
    sxV = xV[:, K].contiguous().clone()
    svm = valid[:, K].contiguous().clone()
    self._s_pre_inA.copy_(self.sample_poisson_spikes_from_analog(sxA, max_rate=300., dt=0.01))
    self._s_pre_inV.copy_(self.sample_poisson_spikes_from_analog(sxV, max_rate=300., dt=0.01))
    self._s_prev_rec.zero_()

    def call_cap():
        sA, sV, sM, sO, dA2M, dV2M = self.update_all_layers_batch(
            sxA, sxV, svm, epoch_idx=epoch_idx, return_delayed=True)
        self._stdp_tail(sA, sV, sM, dA2M, dV2M,
                        self._s_pre_inA, self._s_pre_inV, self._s_prev_rec,
                        epoch_idx, debug=False)

    keys = GDF._l6_complete_keys(self)
    S = GDF._l6_snapshot(self, keys)
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            call_cap()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    GDF._l6_restore(self, S); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call_cap()
    torch.cuda.synchronize()
    return g, sxA, sxV, svm


def _l8_install_full(self, B, epoch_idx):
    self._lever_L4_graph = True
    for nm in ('_g_out_sA', '_g_out_sV', '_g_out_sMSI',
               '_g_out_dA2M', '_g_out_dV2M', '_g_out_sOut'):
        t = getattr(self, nm, None)
        if t is None or t.shape[0] != B:
            setattr(self, nm, torch.zeros((B, self.n), device=self.device))
    ti = getattr(self, '_g_out_sMSI_inh', None)
    if ti is None or ti.shape[0] != B:
        self._g_out_sMSI_inh = torch.zeros((B, self.n_inh), dtype=torch.float32, device=self.device)
    for nm in ('_s_pre_inA', '_s_pre_inV', '_s_prev_rec'):
        t = getattr(self, nm, None)
        if t is None or t.shape[0] != B:
            setattr(self, nm, torch.zeros((B, self.n), device=self.device))
    keys = GDF._l6_complete_keys(self)
    S0 = GDF._l6_snapshot(self, keys)
    RNG0 = GDF._l6_snapshot_rng()
    g, sxA, sxV, svm = _l8_capture_full(self, B, epoch_idx, self.g_rec)
    self._cap_stop_after_substeps = False
    self._cap_stop_after_anchor = False
    self._cap_skip_dbg_sync = False
    GDF._l6_restore(self, S0); torch.cuda.synchronize()
    GDF._l6_restore_rng(RNG0)
    B_cap = B

    def _forward(xA_b, xV_b, valid_mask, pre_inA, pre_inV, epoch_idx=epoch_idx):
        if xA_b.shape[0] != B_cap:
            return False
        sxA.copy_(xA_b); sxV.copy_(xV_b)
        if valid_mask is not None:
            svm.copy_(valid_mask)
        self._s_pre_inA.copy_(pre_inA); self._s_pre_inV.copy_(pre_inV)
        g.replay()                                      # forward + post-loop plasticity + STDP tail
        if epoch_idx > 25:
            self._s_prev_rec.copy_(self._g_out_sMSI)
        return True

    self._l8_forward = _forward
    self._l8_phase = (epoch_idx > 25, B)


# ─────────────────────────────── net builders ───────────────────────────────
def _base_net(seed):
    net = PG.build_net(GDF, seed, batch_size=BS)
    net.plasticity_enabled = True
    net.enable_probe = False
    net._probe = None
    net._panel_enabled = False
    net._lever_L1_drop_inloop_log = False
    net._lever_L2_vector_reset = True
    return net


def build_l6(seed):
    net = _base_net(seed)
    net._lever_L6_graph_train = True
    net._lever_L4_graph = True
    net._lever_L8_fused = False
    return net


def build_l8_partial(seed):
    net = _base_net(seed)
    net._lever_L8_fused = True          # train loop forces L4 on / L6,L7 off; uses in-file PATH-A methods
    return net


def build_l8_full(seed):
    net = _base_net(seed)
    net._lever_L8_fused = True
    net._l8_install = MethodType(_l8_install_full, net)   # TIMING-ONLY: competition in-graph
    return net


def train_one_epoch(net, ep):
    net.g_rec = 0.1 if ep > 25 else 0.0
    net.train_unsupervised_batch(N_SEQ, batch_size=BS, debug=False, epoch_idx=ep)
    net.W_MSI_exc.data.fill_diagonal_(0.0)


def measure_regime(build_fn, eps, tag, log):
    net = build_fn(SEED)
    seed_all(SEED)
    per = []
    for ep in eps:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        train_one_epoch(net, ep)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        per.append(dt)
        print(f"    [{tag} ep{ep:>2}] g_rec={net.g_rec:.1f}  {dt:6.3f}s", flush=True, file=log); log.flush()
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    steady = per[N_DISCARD:]
    return per, sum(steady) / len(steady)


def main():
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ladder_measure.log"), "w", buffering=1)
    hdr = (f"device={dev}  #74 SPEED LADDER  bs={BS} n_seq={N_SEQ} (4x{BS}, no partial)  "
           f"proj80 = 26*g0 + 54*g01   PRE-REG KILL: proj>=600s -> dead-for-<=10min")
    print(hdr, flush=True); print(hdr, flush=True, file=log)

    # global GPU warmup so the very first measured regime is not on cold clocks (throwaway)
    print("[warmup] 2 throwaway L6 epochs to warm GPU clocks ...", flush=True, file=log)
    wn = build_l6(SEED); seed_all(SEED)
    for ep in (0, 1):
        train_one_epoch(wn, ep)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del wn; torch.cuda.empty_cache()

    variants = [("L6", build_l6), ("L8-partial", build_l8_partial), ("L8-full", build_l8_full)]
    rows = {}
    for name, fn in variants:
        print(f"\n[{name}] measuring ...", flush=True); print(f"\n[{name}]", flush=True, file=log)
        per0, m0 = measure_regime(fn, GREC0_EPS, name + "/g0", log)
        per1, m1 = measure_regime(fn, GREC01_EPS, name + "/g01", log)
        proj = 26 * m0 + 54 * m1
        verdict = "DEAD (>=600)" if proj >= 600 else "UNDER (<600)"
        rows[name] = dict(per_g0=per0, per_g01=per1, mean_g0=m0, mean_g01=m1,
                          proj80=proj, verdict=verdict)
        for ln in (f"  g_rec=0   per-ep {['%.2f' % x for x in per0]}  steady mean={m0:.3f}s",
                   f"  g_rec=0.1 per-ep {['%.2f' % x for x in per1]}  steady mean={m1:.3f}s",
                   f"  proj 80-ep = 26*{m0:.3f} + 54*{m1:.3f} = {proj:.1f}s ({proj/60:.2f} min) -> {verdict}"):
            print(ln, flush=True); print(ln, flush=True, file=log)

    sep = "\n" + "=" * 70
    print(sep, flush=True); print(sep, flush=True, file=log)
    head = f"{'variant':<12} {'g0 s/ep':>9} {'g01 s/ep':>9} {'proj80 s':>10} {'min':>7}   verdict"
    print(head, flush=True); print(head, flush=True, file=log)
    for name, _ in variants:
        r = rows[name]
        line = (f"{name:<12} {r['mean_g0']:>9.3f} {r['mean_g01']:>9.3f} "
                f"{r['proj80']:>10.1f} {r['proj80']/60:>7.2f}   {r['verdict']}")
        print(line, flush=True); print(line, flush=True, file=log)
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              "ladder_measure.json"), "w"), indent=1)
    log.close()


if __name__ == "__main__":
    main()
