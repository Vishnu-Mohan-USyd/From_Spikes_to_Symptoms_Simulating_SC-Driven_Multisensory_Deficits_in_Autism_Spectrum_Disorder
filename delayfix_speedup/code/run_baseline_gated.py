#!/usr/bin/env python3
"""#69 BASELINE — a bs=256 ep80 model PROVEN byte-equal to the frozen production delayfix
(md5 5e7d6d20), to serve as the #70 scientific reference + the ~19-min banked fallback.

TWO proofs gate the long run, at the EXACT production config (construct=1000 / mini-batch 256 /
n_seq=1000, so the 256,256,256,232 sequence — incl. the eager 232 partial — is exercised):

  (b) INIT byte-equality : sd_maxdiff(delayfix.build(42), graphdf.build(42)) == 0.0 BEFORE any
                           transfer. Prior gates always transferred delayfix-init in, so they never
                           proved the BARE graphdf init — the very init the 80-epoch mode_a run
                           starts from — equals delayfix's. This proves it.
  (a) PATH byte-fidelity : after 1 production epoch, torch.equal(delayfix, graphdf-L6) at ep0
                           (le25, g_rec=0) AND ep26 (gt25, g_rec=0.1, recurrent STDP active) —
                           reuses the certified train_bitid_gate.compare.

Only if BOTH proofs pass for BOTH epochs do we train 80 epochs (graphdf L6, same config) via the
certified wall_measure.mode_a and save ckpt_ep30 / ckpt_ep79. Any byte-FAIL / segfault -> STOP
(exit 1) and route to the debugger; nothing is papered over. cuda:0 only. delayfix frozen 5e7d6d20;
graphdf 79c45ff. NOTE: graphdf-L6 at construct=1000 / mini=256 is first exercised here — this gate
is exactly its proof (every prior L6 run used construct == mini).
"""
import os, sys, json, importlib
import torch
import faulthandler; faulthandler.enable()
import panelgraph as PG
from train_bitid_gate import compare, train
import wall_measure as WM

DF  = importlib.import_module("Training_delayfix")
GDF = importlib.import_module("Training_graphdf")
SEED, N_SEQ, MINI, CONSTRUCT = 42, 1000, 256, 1000
THIS = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(THIS, "baseline_bs256_gated.log"), "w", buffering=1)


def P(*a):
    print(*a, flush=True)
    print(*a, file=LOG, flush=True)
    LOG.flush()


def microgate(epoch):
    """One production epoch from init at `epoch`'s g_rec branch. Proves (b) init byte-equality
    and (a) delayfix-vs-L6 path byte-fidelity at construct=1000 / mini=256 / n_seq=1000."""
    g_rec = 0.1 if epoch > 25 else 0.0
    base = PG.build_net(DF,  SEED, batch_size=CONSTRUCT)
    opt  = PG.build_net(GDF, SEED, batch_size=CONSTRUCT)
    # (b) INIT byte-equality — measured BEFORE any transfer (the bare graphdf init mode_a uses)
    d0, d0k = PG.sd_maxdiff(base, opt)
    init_ok = (d0 == 0.0)
    P(f"[gate ep{epoch}] (b) INIT  sd_maxdiff(delayfix.build vs graphdf.build, construct={CONSTRUCT}) "
      f"= {d0:.3e} @ {d0k}  -> {'OK (byte-equal init)' if init_ok else 'FAIL: init diverges'}")
    PG.transfer_weights(base, opt)                  # verified no-op when init_ok; matches certified gate
    train(base, SEED, N_SEQ, MINI, epoch, g_rec)                     # delayfix (no levers)
    train(opt,  SEED, N_SEQ, MINI, epoch, g_rec, l1=0, l2=1, l6=1)   # L6: 3 graphed(256) + eager(232) partial
    path_ok = compare(base, opt,
                      f"(a) PATH ep{epoch}  delayfix vs graphdf-L6  "
                      f"(construct={CONSTRUCT} mini={MINI} n_seq={N_SEQ} g_rec={g_rec})")
    del base, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (init_ok and path_ok), init_ok, path_ok


def main():
    P(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}  "
      f"BASELINE gated  seed={SEED}  construct={CONSTRUCT} mini={MINI} n_seq={N_SEQ}")
    P(f"DF ={DF.__file__}\nGDF={GDF.__file__}")

    ok0,  i0,  p0  = microgate(0)
    ok26, i26, p26 = microgate(26)
    P(f"\n{'#'*64}")
    P(f"MICRO-GATE  ep0 : init={'PASS' if i0 else 'FAIL'}  path={'PASS' if p0 else 'FAIL'}")
    P(f"MICRO-GATE  ep26: init={'PASS' if i26 else 'FAIL'}  path={'PASS' if p26 else 'FAIL'}")
    if not (ok0 and ok26):
        P("MICRO-GATE FAILED -> STOP (route to debugger). NOT running the 80-epoch baseline.")
        LOG.close()
        sys.exit(1)
    P("MICRO-GATE PASS — graphdf-L6 byte-equal to delayfix at the production config "
      "(init + le25 + gt25). Running the 80-epoch baseline.\n")

    # ---- 80-epoch baseline: construct=1000 / mini=256 / n_seq=1000, graphdf L6 ----
    WM.BS = MINI
    WM.CONSTRUCT_BS = CONSTRUCT
    res = WM.mode_a(SEED, 80, LOG, save_ckpt=True)
    P(f"\n[BASELINE WALL] 80 ep = {res['wall']:.2f}s ({res['wall']/60:.2f} min)  "
      f"le25={res['le25']:.2f}s gt25={res['gt25']:.2f}s")
    P(f"[BASELINE CKPTS] {res['ckpts']}")
    json.dump(dict(gate_ep0_init=i0, gate_ep0_path=p0, gate_ep26_init=i26, gate_ep26_path=p26,
                   wall=res['wall'], le25=res['le25'], gt25=res['gt25'], ckpts=res['ckpts'],
                   construct=CONSTRUCT, mini=MINI, n_seq=N_SEQ, seed=SEED),
              open(os.path.join(THIS, "baseline_bs256_gated.json"), "w"), indent=1)
    P("BASELINE DONE.")
    LOG.close()


if __name__ == "__main__":
    main()
