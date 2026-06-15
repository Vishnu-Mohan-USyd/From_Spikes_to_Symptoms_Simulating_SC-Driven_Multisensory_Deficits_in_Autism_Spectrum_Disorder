#!/usr/bin/env python3
"""#64 per-epoch BREAKDOWN — decompose the certified-L6 training epoch into:
   * L6-graphed training COMPUTE  (substep-loop graph replay + the eager STDP tail)
   * NON-graphed per-epoch OVERHEAD: data-gen (loc-seq + av-tensor), reset_state, the
     lever-e weight snapshot, and bookkeeping (W_MSI_exc diag-pin)

via NON-INVASIVE runtime monkeypatch timers — the model file (Training_graphdf, 79c45ff) is
NOT edited; we only wrap module-level data-gen fns + the instance reset_state with cuda-synced
timers, then  compute = total - datagen - reset.  Run for a le25 (g_rec=0) and a gt25
(g_rec=0.1) steady epoch.  Tells us whether a byte-identity-preserving OVERHEAD win is on the
table (e.g. data-gen is large) vs. the remaining cost being irreducible graphed compute.
cuda:0 only.
"""
import time, argparse, json
from collections import defaultdict
import numpy as np
import torch
import faulthandler; faulthandler.enable()
import importlib
from wall_measure import build_train_net, seed_all, N_SEQ, BS
from panel_worker import snapshot_weights

GDF = importlib.import_module("Training_graphdf")

T = defaultdict(float); C = defaultdict(int)


def timed(name, fn, sync=False):
    def wrap(*a, **k):
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        r = fn(*a, **k)
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        T[name] += time.time() - t0; C[name] += 1
        return r
    return wrap


# Patch MODULE-level data-gen (train_unsupervised_batch looks these up in module globals at
# call time -> late binding makes the wrapped versions active without touching the model).
GDF.generate_event_loc_seq_batch = timed("datagen_loc", GDF.generate_event_loc_seq_batch, sync=True)
GDF.generate_av_batch_tensor     = timed("datagen_av",  GDF.generate_av_batch_tensor,     sync=True)


def measure_epoch(net, ep):
    orig_reset = net.reset_state
    net.reset_state = timed("reset_state", orig_reset, sync=True)   # instance-shadow
    net.g_rec = 0.1 if ep > 25 else 0.0
    T.clear(); C.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    net.train_unsupervised_batch(N_SEQ, batch_size=BS, debug=False, epoch_idx=ep)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = time.time() - t0
    # bookkeeping: the diag-pin the production loop does every epoch
    tb0 = time.time()
    net.W_MSI_exc.data.fill_diagonal_(0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    book = time.time() - tb0
    # lever-e overhead: the pure-read weight snapshot
    ts0 = time.time()
    _ = snapshot_weights(net)
    snap = time.time() - ts0
    net.reset_state = orig_reset
    datagen = T["datagen_loc"] + T["datagen_av"]
    reset = T["reset_state"]
    compute = total - datagen - reset
    overhead = datagen + reset + snap + book
    return dict(ep=ep, g_rec=net.g_rec, total=total,
                datagen=datagen, datagen_loc=T["datagen_loc"], datagen_av=T["datagen_av"],
                reset=reset, snap=snap, book=book,
                compute=compute, overhead_excl_partial=overhead,
                n_minibatch=C["datagen_av"])


def run_phase(E, seed, n_warm, n_meas):
    net = build_train_net(seed)
    seed_all(seed)
    # warmup epochs at this phase's epoch_idx -> trigger the L6 capture (le25 or gt25)
    for w in range(n_warm):
        ep = E + w
        net.g_rec = 0.1 if ep > 25 else 0.0
        net.train_unsupervised_batch(N_SEQ, batch_size=BS, debug=False, epoch_idx=ep)
        net.W_MSI_exc.data.fill_diagonal_(0.0)
    rows = [measure_epoch(net, E + n_warm + i) for i in range(n_meas)]
    avg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("total", "datagen", "datagen_loc", "datagen_av", "reset",
                     "snap", "book", "compute", "overhead_excl_partial")}
    avg["g_rec"] = rows[0]["g_rec"]; avg["n_minibatch"] = rows[0]["n_minibatch"]
    return avg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--meas", type=int, default=3)
    args = ap.parse_args()
    print(f"device={torch.cuda.get_device_name(0)}  seed={args.seed}  "
          f"warm={args.warm} meas={args.meas}  n_seq={N_SEQ} bs={BS}  "
          f"build=Training_graphdf L1=0/L2=1/L6=1", flush=True)

    out = {}
    for tag, E in (("le25", 0), ("gt25", 26)):
        avg, rows = run_phase(E, args.seed, args.warm, args.meas)
        out[tag] = avg
        print(f"\n=== {tag}  (epoch_idx={E}, g_rec={avg['g_rec']}, "
              f"{int(avg['n_minibatch'])} mini-batches/epoch) ===", flush=True)
        print(f"  TOTAL epoch          : {avg['total']:6.2f}s", flush=True)
        print(f"  ├─ L6 graphed compute: {avg['compute']:6.2f}s   "
              f"({100*avg['compute']/avg['total']:4.1f}%)   [forward replay + eager STDP tail]",
              flush=True)
        print(f"  └─ non-graphed OHEAD : {avg['overhead_excl_partial']:6.2f}s   "
              f"({100*avg['overhead_excl_partial']/avg['total']:4.1f}%)", flush=True)
        print(f"       data-gen        : {avg['datagen']:6.2f}s   "
              f"(loc-seq {avg['datagen_loc']:.2f}s + av-tensor {avg['datagen_av']:.2f}s)", flush=True)
        print(f"       reset_state     : {avg['reset']:6.2f}s", flush=True)
        print(f"       snapshot (lever-e): {avg['snap']*1000:6.1f}ms", flush=True)
        print(f"       bookkeeping     : {avg['book']*1000:6.1f}ms", flush=True)
    json.dump(out, open("epoch_breakdown.json", "w"), indent=1)
    print("\n[wrote epoch_breakdown.json]", flush=True)


if __name__ == "__main__":
    main()
