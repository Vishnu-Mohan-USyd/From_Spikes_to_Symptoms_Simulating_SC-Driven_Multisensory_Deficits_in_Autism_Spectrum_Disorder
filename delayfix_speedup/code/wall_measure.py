#!/usr/bin/env python3
"""#64 lever-e WALL measurement — the MEASURED single-seed 80-epoch TRAINING wall on the
certified L6 build (L1=0, L2=1, L6=1), in three configs (the #53 (a)/(b)/(c) triplet):

  (a) training-only             — no per-epoch panel at all (the FLOOR / decision-critical)
  (b) training + serial panel    — graphed panel-net run INLINE each epoch, RNG saved/restored
                                    (production measure_vitals discipline) -> serializes into wall
  (c) training + overlap panel    — LEVER e: snapshot weights -> separate WORKER process runs the
                                    panel; training never blocks on it -> panel hidden behind train

The training loop mirrors retrain_delayfix.py EXACTLY: build once / seed once, then per epoch
  net.g_rec = 0.1 if ep>25 else 0.0
  net.train_unsupervised_batch(1000, batch_size=256, epoch_idx=ep)
  net.W_MSI_exc.data.fill_diagonal_(0.0)
so the wall is the real production per-epoch cost. A full 80-epoch pass crosses the ep25->26
g_rec gate, exercising BOTH L6 capture phases (le25 at ep0, gt25 at ep26).

Real timed end-to-end runs. cuda:0 only. Pre-registered: full wall <=600s -> ship L6.
"""
import os, sys, time, json, argparse, random
import numpy as np
import torch
import faulthandler; faulthandler.enable()
import importlib
import panelgraph as PG

GDF = importlib.import_module("Training_graphdf")

N_EPOCHS = 80
N_SEQ = 1000     # production per-epoch sequence count (-> 256,256,256,232 mini-batches)
BS = 256
CONSTRUCT_BS = None   # #69 baseline: constructor batch_size (net.batch_size); None => == BS (mini)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def save_rng():
    return (torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            np.random.get_state(), random.getstate())


def restore_rng(st):
    tcpu, tcuda, npst, pyst = st
    torch.set_rng_state(tcpu)
    if tcuda is not None:
        torch.cuda.set_rng_state_all(tcuda)
    np.random.set_state(npst); random.setstate(pyst)


# ───────────────────────── certified L6 training net ─────────────────────────
def build_train_net(seed, bs=None):
    """PG.build_net == retrain_delayfix.build_net recipe (verbatim run_training), on the
    Training_graphdf build, then the CERTIFIED fast lever config: L1=0, L2=1, L6=1."""
    if bs is None:
        bs = BS                              # #69: read module-global at call time so --bs flows through
    construct = CONSTRUCT_BS if CONSTRUCT_BS is not None else bs   # #69 baseline: construct=1000 / mini=256
    net = PG.build_net(GDF, seed, batch_size=construct)
    net.plasticity_enabled = True
    net.enable_probe = False
    net._probe = None
    net._panel_enabled = False
    net._lever_L1_drop_inloop_log = False   # L1 OFF
    net._lever_L2_vector_reset = True        # L2 ON  (certified neutral)
    net._lever_L6_graph_train = True         # L6 ON  (graph the substep loop)
    net._lever_L4_graph = True               # L6 forces the L4 substrate
    return net


def train_one_epoch(net, ep):
    """One production training epoch (mirrors retrain_delayfix.py inner body)."""
    net.g_rec = 0.1 if ep > 25 else 0.0
    net.train_unsupervised_batch(N_SEQ, batch_size=BS, debug=False, epoch_idx=ep)
    net.W_MSI_exc.data.fill_diagonal_(0.0)


VIT_KEYS = [("v_rate", "P5_active_mean_hz"), ("v_EI", "P2B_EI_ratio"),
            ("v_F0", "P1_P_at_0"), ("v_TBW", "P1_width_ms")]


def vitals_from_panel_out(out):
    return {name: (float(out[key]) if out.get(key) is not None else float("nan"))
            for name, key in VIT_KEYS}


# ───────────────────────── (a) training-only ─────────────────────────────────
def mode_a(seed, epochs, log, save_ckpt=False):
    net = build_train_net(seed)
    seed_all(seed)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_eps = {30, epochs - 1} if save_ckpt else set()
    ckpts = {}
    per_ep = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall0 = time.time()
    for ep in range(epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        train_one_epoch(net, ep)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        per_ep.append(dt)
        print(f"[a ep{ep:>2}] g_rec={net.g_rec:.1f}  train={dt:6.2f}s  "
              f"cum={sum(per_ep):7.2f}s", flush=True, file=log)
        log.flush()
        if ep in ckpt_eps:                   # #69: save bs={BS} model for #70 curve-validation (canonical schema)
            ck = GDF.make_checkpoint(net, epoch=ep, comment=f"#69 bs={BS} seed{seed}")
            cpath = os.path.join(this_dir, f"ckpt_ep{ep}_seed{seed}_bs{BS}.pt")
            torch.save(ck, cpath); ckpts[str(ep)] = cpath
            print(f"[a ep{ep:>2}] saved checkpoint {cpath}", flush=True, file=log); log.flush()
    wall = time.time() - wall0
    return dict(mode="a", wall=wall, per_ep=per_ep,
                le25=sum(per_ep[:26]), gt25=sum(per_ep[26:]), ckpts=ckpts)


# ───────────────────────── (b) training + serial panel ───────────────────────
def mode_b(seed, epochs, log):
    net = build_train_net(seed)
    pnet = PG.build_net(GDF, seed, batch_size=540)   # graphed panel-net (B=540 TBW / B=8 volley)
    PG.setup_panel_net(pnet, GDF, 0.0)
    seed_all(seed)
    per_tr, per_pan = [], []
    vitals = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall0 = time.time()
    for ep in range(epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        train_one_epoch(net, ep)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttr = time.time() - t0
        # serial panel, INLINE, RNG saved/restored so training RNG stream is untouched
        rng = save_rng()
        tp0 = time.time()
        PG.transfer_weights(net, pnet)
        pnet.g_rec = net.g_rec
        out, _ = PG.run_panel(pnet, seed, pnet.g_rec)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tpan = time.time() - tp0
        restore_rng(rng)
        per_tr.append(ttr); per_pan.append(tpan)
        vitals.append(dict(ep=ep, **vitals_from_panel_out(out)))
        print(f"[b ep{ep:>2}] g_rec={net.g_rec:.1f}  train={ttr:6.2f}s  panel={tpan:6.2f}s  "
              f"cum={sum(per_tr)+sum(per_pan):7.2f}s", flush=True, file=log)
        log.flush()
    wall = time.time() - wall0
    return dict(mode="b", wall=wall, per_tr=per_tr, per_pan=per_pan, vitals=vitals,
                train_only=sum(per_tr), panel_only=sum(per_pan))


def main():
    global BS, CONSTRUCT_BS
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["a", "b", "c"], required=True)
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=BS,
                    help="#69: train (mini) batch size; 250 => n_seq=1000 = 4 even L6-graphed batches (no eager partial)")
    ap.add_argument("--construct-bs", type=int, default=None,
                    help="#69 baseline: constructor batch_size (net.batch_size); default == --bs. 1000 = production-config byte-equal bs=256 baseline")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="#69: save ckpt_ep30 + ep(final) for #70 curve-validation")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    BS = args.bs
    CONSTRUCT_BS = args.construct_bs

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    tag = args.tag or f"{args.mode}_ep{args.epochs}_s{args.seed}"
    logpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"wall_{tag}.log")
    log = open(logpath, "w", buffering=1)
    hdr = (f"device={dev}  mode={args.mode}  epochs={args.epochs}  seed={args.seed}  "
           f"n_seq={N_SEQ}  bs={BS}  construct_bs={CONSTRUCT_BS if CONSTRUCT_BS is not None else BS}  "
           f"build=Training_graphdf L1=0/L2=1/L6=1")
    print(hdr, flush=True); print(hdr, flush=True, file=log)

    t0 = time.time()
    if args.mode == "a":
        res = mode_a(args.seed, args.epochs, log, save_ckpt=args.save_ckpt)
    elif args.mode == "b":
        res = mode_b(args.seed, args.epochs, log)
    else:
        from panel_worker import mode_c
        res = mode_c(args.seed, args.epochs, log, build_train_net, train_one_epoch,
                     save_rng, restore_rng, vitals_from_panel_out, seed_all)
    res["elapsed_total"] = time.time() - t0

    summary = (f"\n{'='*60}\n[WALL mode {args.mode}] epochs={args.epochs}  "
               f"TOTAL WALL = {res['wall']:.2f}s  ({res['wall']/60:.2f} min)  "
               f"vs 600s budget -> {'UNDER' if res['wall'] <= 600 else 'OVER'}")
    print(summary, flush=True); print(summary, flush=True, file=log)
    json.dump(res, open(logpath.replace(".log", ".json"), "w"), indent=1)
    log.close()


if __name__ == "__main__":
    main()
