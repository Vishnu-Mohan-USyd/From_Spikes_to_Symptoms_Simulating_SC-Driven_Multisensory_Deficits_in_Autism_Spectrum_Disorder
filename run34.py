#!/usr/bin/env python3
"""TASK #34 — full ep0->79 single-seed identity driver: canonical(old) vs L1+L2+L4(new).

Cross-process design (per Lead): each build runs in its own process from the SAME
seed and saves a per-epoch state_dict; cmp34.py compares them. state_dict has 19
entries (13 params + 6 const buffers); W_msi2out + all out-layer dynamics are plain
attrs (NOT registered) -> the #40 readout non-determinism is auto-excluded.

Identity rests on FOUR verified facts:
  (1) update_all_layers_batch draws NO RNG  -> graph-replay stays RNG-locked to eager.
  (2) reset_state draws NO RNG              -> the STEP-A cache (in-place restore,
                                               skips the normal reset) cannot desync
                                               the global RNG vs canonical's full reset.
  (3) levers touch only forward/reset/attrs -> init + data-gen are RNG-identical.
  (4) the captured substep loop bakes the two epoch-gated branches (epoch_idx==5 has
      a .item() host-sync => MUST run eager; epoch_idx>25 toggles in-loop iSTDP +
      recurrent block + msi-anchor) and g_rec (0.0 le25 / 0.1 gt25).
  => exactly TWO phase graphs {le25, gt25} at B=256, ep5 eager, B constant (n_seq a
     multiple of 256, no tail) so the up-front graphs' baked addresses stay valid.

The L4 capture draws RNG for its throwaway sample inputs; we snapshot/restore ALL
RNG state around it so the build is RNG-transparent (eager build never captures, so
both builds enter the epoch loop with identical RNG).

Modes:
  --mode canon  : canonical Training.py (md5 466c9a76), no levers, no graph.
  --mode l4off  : Training_graph.py, L1+L2 on, L4 OFF (pure eager == the ship).
  --mode l4on   : Training_graph.py, L1+L2+L4 on (phase graphs + ep5 eager).
  --selfcheck   : same-process l4off-vs-l4on, reseed-per-epoch, in-memory per-epoch
                  compare (fast cheap-gate of the phase-dispatch logic).

5090: CUDA_VISIBLE_DEVICES=0 python -u run34.py --mode l4on --out <dir> --epochs N --nseq M
"""
import os, sys, math, time, copy, json, argparse, importlib.util
import numpy as np
import torch
import random

ROOT = os.path.dirname(os.path.abspath(__file__))  # flat
CODE = ROOT
GRAPHPATH = os.path.join(ROOT, "Training_graphdf_d52.py")
sys.path.insert(0, CODE)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


import Training as CANON                       # pristine canonical (466c9a76)
GRAPH = load_mod("Training_graph", GRAPHPATH)   # L1+L2+L4 sandbox

# spike handles the wrapper returns (positions: sA,sV,sMSI,sOut,dA2M,dV2M)
SPK = ["_latest_sA", "_latest_sV", "_latest_sMSI",
       "_latest_sOut", "_latest_dA2M", "_latest_dV2M"]


# ───────────────────────── init (verbatim run_training build) ─────────────────
def init_net(MOD, batch_size, seed):
    """Replicates run_training()'s net build+calibration up to the epoch loop.
    Same code for every module -> RNG-identical init (verified by ep_init compare)."""
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
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
        net.gNMDA = 1.30; net.tau_nmda = 80.0; net.nmda_alpha = 0.1
        net.Erev_nmda = 20.0; net.tau_nmdaVolt = 100.0; net.v_nmda_rest = -65.0
        net.nmda_vrest_offset = 7.0; net.mg_vhalf = -35.0; net.tau_nmda_inh = 21.6
    net.u_a.fill_(0.7); net.u_v.fill_(0.7); net.tau_rec = 400.0
    net.input_scaling = 400; net.g_GABA = 10
    return net


# ───────────────────────── complete snapshot / restore ───────────────────────
def complete_keys(net):
    keys = []
    for k, v in list(vars(net).items()):
        if torch.is_tensor(v) and v.is_floating_point():
            keys.append(("attr", k))
    for k, v in net.named_parameters():
        keys.append(("param", k))
    for k, v in net.named_buffers():
        if v is not None and torch.is_tensor(v) and v.is_floating_point():
            keys.append(("buf", k))
    return keys


def _resolve(net, kind, name):
    if kind == "attr":
        return getattr(net, name)
    obj = net; *parents, last = name.split(".")
    for p in parents:
        obj = getattr(obj, p)
    return getattr(obj, last)


def snapshot(net, keys):
    snap, seen = {}, set()
    for kind, name in keys:
        t = _resolve(net, kind, name)
        if not torch.is_tensor(t) or id(t) in seen:
            continue
        seen.add(id(t)); snap[(kind, name)] = t.detach().clone()
    snap[("_pos", "_delay_positions")] = copy.deepcopy(net._delay_positions)
    snap[("_gpos", "g")] = {s: getattr(net, '_gpos_' + s).clone()
                            for _b, s in net._GPOS_NAMES}
    return snap


def restore(net, snap):
    net._delay_positions = copy.deepcopy(snap[("_pos", "_delay_positions")])
    for s, v in snap[("_gpos", "g")].items():
        getattr(net, '_gpos_' + s).copy_(v)
    for (kind, name), val in snap.items():
        if kind in ("_pos", "_gpos"):
            continue
        t = _resolve(net, kind, name)
        if torch.is_tensor(t) and t.shape == val.shape:
            t.copy_(val)


def snapshot_rng():
    return (torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            np.random.get_state(), random.getstate())


def restore_rng(s):
    torch.set_rng_state(s[0])
    if s[1] is not None:
        torch.cuda.set_rng_state_all(s[1])
    np.random.set_state(s[2]); random.setstate(s[3])


# ───────────────────────── phase-graph capture ───────────────────────────────
def capture_phase(net, MOD, epoch_idx_repr, g_rec, B):
    """Capture the LOOP-ONLY substep graph for one phase. RNG-DIRTY (sample inputs);
    the caller wraps this in an RNG snapshot/restore. Mutates net weights via warmup
    (the caller restores the complete pre-capture state afterwards)."""
    net.g_rec = g_rec
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
    net._cap_stop_after_substeps = False           # warm FULL fwd -> builds caches
    for t in range(K):
        net.update_all_layers_batch(xA[:, t], xV[:, t], valid[:, t],
                                    epoch_idx=epoch_idx_repr, return_delayed=True)

    sxA = xA[:, K].contiguous().clone()
    sxV = xV[:, K].contiguous().clone()
    svm = valid[:, K].contiguous().clone()

    net._cap_stop_after_substeps = True            # *** capture LOOP ONLY ***
    def call_cap():
        return net.update_all_layers_batch(sxA, sxV, svm,
                                           epoch_idx=epoch_idx_repr, return_delayed=True)

    keys = complete_keys(net)
    S = snapshot(net, keys)
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            call_cap()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    restore(net, S); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call_cap()
    torch.cuda.synchronize()
    return g, sxA, sxV, svm


def make_phase_wrapper(net, MOD, ORIG, graphs):
    """epoch_idx==5 -> eager (ORIG); else replay le25/gt25 graph + EAGER plasticity
    tail. Returns the PERSISTENT static output buffers (net._g_out_*) that the captured
    region writes in-place — never the escaped graph-internal _latest_* tensors."""
    B_cap = graphs["le25"][1].shape[0]      # captured batch size (256)
    def wrapper(xA_b, xV_b, valid_mask=None, epoch_idx=0, return_delayed=True, **kw):
        # ep5 carries a host-sync .item() (must run eager); the canonical 1000-cadence
        # B=232 tail isn't the captured shape -> run it EAGER on its own persistent pool
        # (bit-identical by construction). Only full B=256 batches replay the graph.
        if epoch_idx == 5 or xA_b.shape[0] != B_cap:
            return ORIG(xA_b, xV_b, valid_mask, epoch_idx=epoch_idx,
                        return_delayed=return_delayed, **kw)
        g, sxA, sxV, svm = graphs["gt25" if epoch_idx > 25 else "le25"]
        sxA.copy_(xA_b); sxV.copy_(xV_b)
        if valid_mask is not None:
            svm.copy_(valid_mask)
        g.replay()
        # The eager plasticity tail reads net._latest_sA/_latest_sV/_latest_sMSI
        # (apply_topographic_anchor_unimodal, apply_local_competition_*). Those attrs are
        # ESCAPED graph-internal tensors whose addresses gt25's capture reallocates, so
        # after replay they hold stale spikes. Repoint them at the PERSISTENT _g_out_*
        # buffers (which the captured loop fills in-place every replay) so the tail reads
        # this frame's true final-substep spikes — the actual two-graph corruption fix.
        net._latest_sA = net._g_out_sA
        net._latest_sV = net._g_out_sV
        net._latest_sMSI = net._g_out_sMSI
        net._latest_sOut = net._g_out_sOut
        net._latest_dA2M = net._g_out_dA2M
        net._latest_dV2M = net._g_out_dV2M
        if getattr(net, 'plasticity_enabled', True):
            MOD.apply_topographic_anchor_unimodal(net, layer="A", lr=1.0 * net.lr_uni, sigma=2.5)
            MOD.apply_topographic_anchor_unimodal(net, layer="V", lr=1.0 * net.lr_uni, sigma=2.5)
            MOD.apply_local_competition_unimodal_fast(net, "A", beta=2.0 * net.lr_uni, neighbour_dist=4)
            MOD.apply_local_competition_unimodal_fast(net, "V", beta=2.0 * net.lr_uni, neighbour_dist=4)
            if epoch_idx > 25:
                MOD.apply_local_competition_msi_fast(net, beta=1.5 * net.lr_msi, neighbour_dist=6)
            MOD.soft_row_scaling(net)
        return (net._g_out_sA, net._g_out_sV, net._g_out_sMSI,
                net._g_out_sOut, net._g_out_dA2M, net._g_out_dV2M)
    return wrapper


def install_l4(net, MOD):
    """Capture both phase graphs (RNG-transparent + state-transparent) and monkeypatch
    net.update_all_layers_batch to the phase-dispatch wrapper. Net is unchanged on
    return except for the monkeypatch (weights/dynamics/RNG fully restored)."""
    net._lever_L4_graph = True
    B = 256
    net.reset_state(B)              # materialize _gpos_* + build the STEP-A cache
    # persistent STATIC output buffers (stable addresses, shared by both phase graphs);
    # the captured loop writes the final-substep spikes into these in-place so nothing
    # escapes the capture (fixes the two-graph stale-output corruption).
    for nm in ('_g_out_sA', '_g_out_sV', '_g_out_sMSI', '_g_out_dA2M', '_g_out_dV2M', '_g_out_sOut'):
        setattr(net, nm, torch.zeros((B, net.n), device=net.device))
    keys = complete_keys(net)       # (RNG-free; weights untouched -> S0 keeps fresh init)
    S0 = snapshot(net, keys)
    RNG0 = snapshot_rng()
    graphs = {}
    graphs["le25"] = capture_phase(net, MOD, epoch_idx_repr=10, g_rec=0.0, B=B)
    if os.environ.get("FSTS_R34_LE25ONLY", "0") != "1":
        graphs["gt25"] = capture_phase(net, MOD, epoch_idx_repr=26, g_rec=0.1, B=B)
    # restore eager-path flags so the ep5 fallback runs the FULL method (not loop-only)
    net._cap_stop_after_substeps = False
    net._cap_skip_dbg_sync = False
    restore(net, S0); torch.cuda.synchronize()
    restore_rng(RNG0)
    ORIG = net.update_all_layers_batch                     # bound method (ep5 eager)
    net.update_all_layers_batch = make_phase_wrapper(net, MOD, ORIG, graphs)
    return graphs


# ───────────────────────── vitals + state_dict io ────────────────────────────
def sd_cpu(net):
    return {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}


def vitals(net, sd):
    nan = any(torch.isnan(v).any().item() for v in sd.values())
    def g(n):
        return sd[n] if n in sd else None
    def mx(n):
        t = g(n); return float(t.abs().max()) if t is not None else float('nan')
    def sm(n):
        t = g(n); return float(t.sum()) if t is not None else float('nan')
    return dict(
        nan=bool(nan),
        max_W_inA=mx("W_inA"), max_W_a2msi_AMPA=mx("W_a2msi_AMPA"),
        max_W_v2msi_AMPA=mx("W_v2msi_AMPA"), sum_W_msiInh2Exc_GABA=sm("W_msiInh2Exc_GABA"),
        sum_W_MSI_exc=sm("W_MSI_exc"),
        W_msi2out_max=float(net.W_msi2out.abs().max()) if hasattr(net, "W_msi2out") else float('nan'),
    )


# ───────────────────────── epoch loop ────────────────────────────────────────
def run_epochs(net, n_epochs, n_seq, out_dir, label, save=True):
    os.makedirs(out_dir, exist_ok=True) if (save and out_dir) else None
    if save and out_dir:
        torch.save(sd_cpu(net), os.path.join(out_dir, "sd_epinit.pt"))
    vit_rows = []
    print(f"[{label}] {'epoch':>5} {'phase':>5} {'s/epoch':>9}  nan  max|W_inA|")
    for ep in range(n_epochs):
        net.g_rec = 0.1 if ep > 25 else 0.0
        torch.cuda.synchronize(); t0 = time.time()
        net.train_unsupervised_batch(n_seq, batch_size=256, debug=False, epoch_idx=ep)
        torch.cuda.synchronize(); dt = time.time() - t0
        sd = sd_cpu(net)
        if save and out_dir:
            torch.save(sd, os.path.join(out_dir, f"sd_ep{ep}.pt"))
        v = vitals(net, sd); v.update(epoch=ep, s=dt,
                                      phase=("ep5" if ep == 5 else "gt25" if ep > 25 else "le25"))
        vit_rows.append(v)
        print(f"[{label}] {ep:>5} {v['phase']:>5} {dt:>9.3f}  {int(v['nan'])}   {v['max_W_inA']:.4f}")
    if save and out_dir:
        with open(os.path.join(out_dir, "vitals.json"), "w") as f:
            json.dump(vit_rows, f, indent=1)
    return vit_rows


def speed_summary(rows, label):
    le = [r["s"] for r in rows if r["phase"] == "le25"][1:]   # drop ep0 warm-in
    gt = [r["s"] for r in rows if r["phase"] == "gt25"]
    le_m = sum(le) / len(le) if le else float('nan')
    gt_m = sum(gt) / len(gt) if gt else float('nan')
    print(f"[{label}] mean s/epoch  le25={le_m:.3f}  gt25={gt_m:.3f}")
    return le_m, gt_m


# ───────────────────────── modes ─────────────────────────────────────────────
def diff_sd(a, b):
    worst, who = 0.0, None
    for k in a:
        if k not in b:
            continue
        d = (a[k].double() - b[k].double()).abs().max().item()
        if d > worst:
            worst, who = d, k
    return worst, who


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["canon", "l4off", "l4on"], default=None)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--nseq", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # 1000-cadence = 256x3 + 232 tail; the B=232 tail routes EAGER (graph is B=256 only),
    # both pools persistent so the graph survives the intervening tail. No multiple-of-256
    # constraint anymore.
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    if args.selfcheck:
        print(f"[sc] device={dev}  SELF-CHECK l4off-vs-l4on  epochs={args.epochs} nseq={args.nseq}")
        net_off = init_net(GRAPH, 256, args.seed); net_off._lever_L4_graph = False
        net_on = init_net(GRAPH, 256, args.seed)
        install_l4(net_on, GRAPH)
        # both built from seed -> identical post-init (incl W_msi2out); verify:
        w0, who0 = diff_sd(sd_cpu(net_off), sd_cpu(net_on))
        print(f"[sc] post-init state_dict max|Δ| = {w0:.3e}  ({who0})")
        worst = 0.0
        print(f"[sc] {'epoch':>5} {'phase':>5} {'max|Δsd|':>11} {'off s':>8} {'on s':>8} {'spd':>6}")
        roff, ron = [], []
        for ep in range(args.epochs):
            s = 4200 + ep
            net_off.g_rec = net_on.g_rec = (0.1 if ep > 25 else 0.0)
            torch.manual_seed(s); np.random.seed(s); random.seed(s); torch.cuda.manual_seed_all(s)
            torch.cuda.synchronize(); t0 = time.time()
            net_off.train_unsupervised_batch(args.nseq, batch_size=256, debug=False, epoch_idx=ep)
            torch.cuda.synchronize(); toff = time.time() - t0
            torch.manual_seed(s); np.random.seed(s); random.seed(s); torch.cuda.manual_seed_all(s)
            torch.cuda.synchronize(); t0 = time.time()
            net_on.train_unsupervised_batch(args.nseq, batch_size=256, debug=False, epoch_idx=ep)
            torch.cuda.synchronize(); ton = time.time() - t0
            d, who = diff_sd(sd_cpu(net_off), sd_cpu(net_on))
            worst = max(worst, d)
            ph = "ep5" if ep == 5 else "gt25" if ep > 25 else "le25"
            roff.append(dict(s=toff, phase=ph)); ron.append(dict(s=ton, phase=ph))
            print(f"[sc] {ep:>5} {ph:>5} {d:>11.3e} {toff:>8.3f} {ton:>8.3f} {toff/ton:>5.2f}x"
                  + (f"  <- {who}" if d > 0 else ""))
        print()
        speed_summary(roff, "sc-off"); speed_summary(ron, "sc-on ")
        if worst == 0.0:
            print(f"\n[VERDICT] SELF-CHECK PASS — l4off vs l4on bit-identical over {args.epochs} epochs "
                  f"(crosses ep5 eager + ep25->26 phase flip). Phase-dispatch driver is identity-clean.")
        else:
            print(f"\n[VERDICT] SELF-CHECK FAIL — max|Δsd|={worst:.3e} -> STOP, report to Lead.")
        return

    assert args.mode and args.out, "single-build run needs --mode and --out"
    MOD = CANON if args.mode == "canon" else GRAPH
    print(f"[{args.mode}] device={dev}  out={args.out}  epochs={args.epochs} nseq={args.nseq} seed={args.seed}")
    # build at the TRAINING batch size (256), not nseq: a wider init then reset(256) would
    # FREE a big dynamics buffer set whose blocks the caching allocator can hand back over
    # the graph's baked addresses (segfault on replay). 256 == the per-mini-batch size.
    net = init_net(MOD, 256, args.seed)
    if args.mode == "l4off":
        net._lever_L4_graph = False
    elif args.mode == "l4on":
        install_l4(net, GRAPH)
    rows = run_epochs(net, args.epochs, args.nseq, args.out, args.mode, save=True)
    le_m, gt_m = speed_summary(rows, args.mode)
    print(f"[{args.mode}] DONE — wrote {args.epochs} per-epoch state_dicts to {args.out}")


if __name__ == "__main__":
    main()
