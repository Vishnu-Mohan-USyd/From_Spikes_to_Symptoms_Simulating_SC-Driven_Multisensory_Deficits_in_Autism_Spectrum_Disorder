#!/usr/bin/env python3
"""#69 GATE — prove the FUSED single-graph build (lever L8) is byte-identical to the un-fused
bs=250 build (lever L6) AND to the frozen delayfix reference, at the production-equivalent bs=250
config (n_seq=1000 / mini=250 / construct=250 -> 4 even B=250 batches, NO partial), after ONE real
train epoch — at ep0 (le25, g_rec=0, 6 FF gers) AND ep26 (gt25, g_rec=0.1, + recurrent MSI ger +
W_MSI_exc fill_diagonal). torch.equal (byte-for-byte), NOT allclose.

Three nets, all started byte-identical (delayfix init transferred in):
  ref = Training_delayfix, eager (no levers)                          — the frozen reference at mini=250
  l6  = Training_graphdf,  L6 (un-fused fwd graph + EAGER STDP tail)  — the bs=250 floor build
  l8  = Training_graphdf,  L8 (FUSED fwd + post-loop plasticity + STDP single graph) — the SUBJECT

PASS requires, at BOTH epochs: ref==l8 (fundamental) AND l6==l8 (the un-fused-equivalence the Lead
named) AND ref==l6 (sanity). ANY byte-FAIL / segfault -> STOP (exit 1), route to the debugger;
nothing papered over. cuda:0 only; delayfix frozen 5e7d6d20.

NOTE — the make-or-break risk this gate settles: the L8 capture folds the post-loop plasticity
(apply_local_competition_unimodal_fast's .median(), soft_row_scaling's .norm()) INTO the graph. The
L6 author deliberately captured the substep loop ONLY and hedged those reductions "may not be
stream-capture-safe". This gate is exactly the test of whether they capture byte-identically (PASS)
or not (capture error / segfault / byte-FAIL -> debugger).
"""
import os, sys, json, importlib
import torch
import faulthandler; faulthandler.enable()   # dump a Python traceback if a capture/replay segfaults
import panelgraph as PG
from train_bitid_gate import compare, train

DF  = importlib.import_module("Training_delayfix")
GDF = importlib.import_module("Training_graphdf")
SEED, N_SEQ, BS = 42, 1000, 250            # 4 even B=250 batches, NO eager partial
THIS = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(THIS, "gate_fused.log"), "w", buffering=1)


def P(*a):
    print(*a, flush=True)
    print(*a, file=LOG, flush=True)
    LOG.flush()


def gate(epoch):
    """One real train epoch from init at `epoch`'s g_rec branch; proves ref/l6/l8 are mutually
    byte-identical at bs=250 (4x250, no partial)."""
    g_rec = 0.1 if epoch > 25 else 0.0
    ref = PG.build_net(DF,  SEED, batch_size=BS)
    l6  = PG.build_net(GDF, SEED, batch_size=BS)
    l8  = PG.build_net(GDF, SEED, batch_size=BS)
    # INIT byte-equality (the bare init each net trains from), measured BEFORE any transfer
    d6, d6k = PG.sd_maxdiff(ref, l6)
    d8, d8k = PG.sd_maxdiff(ref, l8)
    P(f"[gate ep{epoch}] INIT  ref==l6 max|Δ|={d6:.3e}@{d6k}   ref==l8 max|Δ|={d8:.3e}@{d8k}  "
      f"-> {'OK' if (d6 == 0.0 and d8 == 0.0) else 'FAIL: init diverges'}")
    for dst in (l6, l8):
        PG.transfer_weights(ref, dst)               # belt-and-suspenders (verified no-op when init Δ=0)
    l8._lever_L8_fused = True                        # arm the FUSED single graph (its loop forces L4, off L6/L7)

    train(ref, SEED, N_SEQ, BS, epoch, g_rec)                    # delayfix eager (no levers)
    train(l6,  SEED, N_SEQ, BS, epoch, g_rec, l1=0, l2=1, l6=1)  # un-fused: L6 fwd graph + EAGER STDP tail
    train(l8,  SEED, N_SEQ, BS, epoch, g_rec, l1=0, l2=1)        # FUSED: L8 (loop forces L4, disables L6/L7)

    ok_rl8  = compare(ref, l8, f"FUSED ep{epoch}  ref(delayfix) == l8(L8 fused)         bs={BS} g_rec={g_rec}")
    ok_l6l8 = compare(l6,  l8, f"FUSED ep{epoch}  l6(un-fused)  == l8(L8 fused)         bs={BS} g_rec={g_rec}")
    ok_rl6  = compare(ref, l6, f"FUSED ep{epoch}  ref(delayfix) == l6(un-fused) sanity  bs={BS} g_rec={g_rec}")
    del ref, l6, l8
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (ok_rl8 and ok_l6l8 and ok_rl6), dict(
        ref_l8=ok_rl8, l6_l8=ok_l6l8, ref_l6=ok_rl6,
        init_ref_l6=(d6 == 0.0), init_ref_l8=(d8 == 0.0))


def main():
    P(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}  "
      f"FUSED-GRAPH gate  seed={SEED}  bs={BS} n_seq={N_SEQ}  (4x{BS}, no partial)")
    P(f"DF ={DF.__file__}\nGDF={GDF.__file__}")

    ok0,  d0  = gate(0)
    ok26, d26 = gate(26)
    P(f"\n{'#'*64}")
    P(f"FUSED-GATE ep0 : {d0}")
    P(f"FUSED-GATE ep26: {d26}")
    allok = ok0 and ok26
    if allok:
        P("FUSED-GATE PASS — L8 (fused single graph) is byte-identical to the un-fused bs=250 "
          "build AND to delayfix, at ep0 (le25) AND ep26 (gt25). The post-loop plasticity "
          "(.median()/.norm()) captures cleanly.")
    else:
        P("FUSED-GATE FAIL -> STOP. NOT clearing L8 for the fused retrain. Route to the debugger "
          "with the diverging tensors / capture traceback. No papering.")
    json.dump(dict(ep0=d0, ep26=d26, passed=allok, bs=BS, n_seq=N_SEQ, seed=SEED),
              open(os.path.join(THIS, "gate_fused.json"), "w"), indent=1)
    LOG.close()
    if not allok:
        sys.exit(1)


if __name__ == "__main__":
    main()
