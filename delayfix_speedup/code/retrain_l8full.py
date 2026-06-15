#!/usr/bin/env python3
"""#75 PATH B — REAL L8-full bs=250 retrain (the <=10-min candidate).

Builds the FUSED-FULL single-graph build (lever L8-full: the WHOLE per-ext-step body — substeps +
post-loop plasticity INCLUDING the in-graph unimodal-competition matmul + the STDP tail — captured
as ONE CUDA graph) and runs the production training protocol ep0->79, seed 42, bs=250.

Protocol is BYTE-PROTOCOL-IDENTICAL to the certified bs=250 / baseline lineage (wall_measure.py
mode_a == retrain_delayfix.py inner body) so the validator's downstream TBW/SBW/E-I re-val has its
common-mode cancel — the ONLY intended difference vs the certified bs=250 build is L6 -> L8-full,
i.e. the proven (#73) competition-GEMM kernel-selection 1.58e-4 W_in drift the user pre-authorized
("small weight change OK for <=10 if TBW/SBW/E-I stay similar"). Everything else (seed-once-then-
loop, g_rec=0.1 if ep>25 else 0.0, n_seq=1000, W_MSI_exc.fill_diagonal_(0) each epoch, L1=0/L2=1)
is identical.

Deliverable = the MEASURED 80-epoch wall (synchronize-bracketed, real number — NOT the 5.89-min
ladder projection) + ckpt_ep30 / ckpt_ep79 saved to DISTINCT *_L8full.pt names so the banked
bs=250 fallback is NOT clobbered.

Rolling smoke-gate every ~10 ep (+ final): plastic-weight norms finite & not >10x their init, an
MSI firing-rate proxy, and an all-weights NaN/Inf sweep. KILL (stop, exit 1, no ckpt) on any NaN/Inf
or a >10x-init weight-norm blow-up. This is a vitals smoke-gate ONLY — the scientific TBW/SBW/E-I
equivalence re-val is the validator's job on the ep79 ckpt, NOT run here.

cuda:0 only (serial). Training_delayfix.py FROZEN (md5 5e7d6d20) — untouched; all edits live in the
opt build (Training_graphdf.py).
"""
import os, sys, time, json, random, importlib
import numpy as np
import torch
import faulthandler; faulthandler.enable()   # dump a Python traceback if a capture/replay segfaults
import panelgraph as PG

GDF = importlib.import_module("Training_graphdf")

SEED      = 42
BS        = 250          # even divisor of n_seq=1000 -> 4 x 250, NO eager partial
N_SEQ     = 1000
N_EPOCHS  = 80
THIS      = os.path.dirname(os.path.abspath(__file__))
CKPT_EP30 = os.path.join(THIS, "ckpt_ep30_seed42_bs250_L8full.pt")
CKPT_EP79 = os.path.join(THIS, "ckpt_ep79_seed42_bs250_L8full.pt")
GATE_EPS  = set(range(0, N_EPOCHS, 10)) | {N_EPOCHS - 1}   # 0,10,...,70, 79
NORM_BLOWUP = 10.0       # KILL if any monitored weight norm exceeds 10x its init

# plastic weights to hard-monitor (norm vs init); the in-graph competition touches W_inA/W_inV (the
# accepted 1.58e-4 drift), the STDP tail touches the rest. NaN/Inf is swept over ALL of them.
MON_W = ["W_inA", "W_inV", "W_MSI_exc", "W_MSI_inh", "W_inA_inh", "W_inV_inh",
         "W_a2msi_AMPA", "W_a2msi_NMDA", "W_v2msi_AMPA", "W_v2msi_NMDA"]


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ───────────────────────── L8-full training net ─────────────────────────
def build_l8full_net(seed):
    """PG.build_net == retrain_delayfix.build_net recipe on the Training_graphdf build, then the
    CERTIFIED fast config (L1=0, L2=1) + the PATH B L8-FULL lever cascade. construct=mini=250 so
    n_seq=1000 = 4 even B=250 batches (no eager partial), exactly like the certified bs=250 gate."""
    net = PG.build_net(GDF, seed, batch_size=BS)
    net.plasticity_enabled = True
    net.enable_probe = False
    net._probe = None
    net._panel_enabled = False
    net._lever_L1_drop_inloop_log = False    # L1 OFF  (certified)
    net._lever_L2_vector_reset    = True     # L2 ON   (certified neutral)
    # PATH B L8-full cascade (mirrors the in-__init__ env-var cascade explicitly so it does not
    # depend on env-var-before-construction timing):
    net._lever_L8_full      = True           # capture the WHOLE body incl the competition matmul
    net._lever_L8_fused     = True           # L8-full is a MODE of L8 -> single fused graph
    net._lever_L4_graph     = True           # L8 needs the L4 substrate (gpos + alloc-free reset)
    net._lever_L6_graph_train = False        # L8 REPLACES the L6 forward graph
    net._lever_L7_graph_stdp  = False        # L8 SUBSUMES the L7 STDP-tail graph (one graph only)
    return net


def train_one_epoch(net, ep):
    """One production training epoch — byte-protocol-identical to wall_measure.mode_a /
    retrain_delayfix.py inner body."""
    net.g_rec = 0.1 if ep > 25 else 0.0
    net.train_unsupervised_batch(N_SEQ, batch_size=BS, debug=False, epoch_idx=ep)
    net.W_MSI_exc.data.fill_diagonal_(0.0)


# ───────────────────────── rolling smoke-gate vitals ─────────────────────────
def weight_norms(net):
    d = {}
    for nm in MON_W:
        w = getattr(net, nm, None)
        if w is None:
            continue
        w = w.data if hasattr(w, "data") else w
        d[nm] = float(w.norm().item())
    return d


def all_weights_finite(net):
    """Sweep every monitored plastic weight + the graph output buffers for NaN/Inf."""
    bad = []
    for nm in MON_W + ["_g_out_sA", "_g_out_sV", "_g_out_sMSI", "_g_out_dA2M", "_g_out_dV2M"]:
        w = getattr(net, nm, None)
        if w is None:
            continue
        w = w.data if hasattr(w, "data") else w
        if not bool(torch.isfinite(w).all().item()):
            bad.append(nm)
    return bad


def msi_rate(net):
    """MSI firing-rate proxy = mean of the last replay's MSI spike-output buffer."""
    m = getattr(net, "_g_out_sMSI", None)
    if m is None:
        return float("nan")
    return float(m.float().mean().item())


def smoke_gate(net, ep, init_norms, log):
    """Return (ok, msg). KILL (ok=False) on any NaN/Inf or a >10x-init weight-norm blow-up."""
    bad = all_weights_finite(net)
    norms = weight_norms(net)
    rate = msi_rate(net)
    blown = [(nm, norms[nm], init_norms[nm])
             for nm in norms
             if init_norms.get(nm, 0.0) > 0 and norms[nm] > NORM_BLOWUP * init_norms[nm]]
    ratios = {nm: (norms[nm] / init_norms[nm] if init_norms.get(nm, 0.0) > 0 else float("nan"))
              for nm in norms}
    ratio_str = "  ".join(f"{nm}={ratios[nm]:.3f}x" for nm in MON_W if nm in ratios)
    P(log, f"[gate ep{ep:>2}] msi_rate={rate:.4e}  finite={'OK' if not bad else 'BAD:'+','.join(bad)}")
    P(log, f"[gate ep{ep:>2}] norm/init: {ratio_str}")
    if bad:
        return False, f"NaN/Inf in {bad} at ep{ep}"
    if blown:
        return False, ("weight-norm blow-up at ep%d: " % ep +
                       "; ".join(f"{nm} {n:.3e} > {NORM_BLOWUP}x init {i:.3e}" for nm, n, i in blown))
    return True, "ok"


def P(log, *a):
    print(*a, flush=True)
    print(*a, file=log, flush=True)
    log.flush()


# ───────────────────────── main retrain ─────────────────────────
def main():
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log = open(os.path.join(THIS, "retrain_l8full.log"), "w", buffering=1)
    P(log, f"device={dev}  #75 PATH B  L8-FULL retrain  seed={SEED}  bs={BS}  n_seq={N_SEQ}  "
           f"epochs={N_EPOCHS}  (4x{BS}, no partial)")
    P(log, f"GDF={GDF.__file__}")
    P(log, f"schedule: g_rec=0.0 for ep<26, 0.1 for ep>=26   ckpts -> {os.path.basename(CKPT_EP30)} / "
           f"{os.path.basename(CKPT_EP79)}")
    P(log, f"smoke-gate: every ~10 ep + final; KILL on NaN/Inf or weight-norm > {NORM_BLOWUP}x init")

    # build (seeded by PG.build_net) then seed-once before the loop — EXACT mode_a / retrain_delayfix
    # protocol so the re-val common-mode cancels.
    net = build_l8full_net(SEED)
    seed_all(SEED)
    init_norms = weight_norms(net)
    P(log, f"[init] weight norms: " + "  ".join(f"{nm}={init_norms[nm]:.4e}" for nm in MON_W if nm in init_norms))

    per_ep = []
    ckpts = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall0 = time.time()
    killed = None
    for ep in range(N_EPOCHS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        train_one_epoch(net, ep)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        per_ep.append(dt)
        P(log, f"[ep{ep:>2}] g_rec={net.g_rec:.1f}  train={dt:6.2f}s  cum={sum(per_ep):8.2f}s "
               f"({sum(per_ep)/60:5.2f}min)")

        if ep in GATE_EPS:
            ok, msg = smoke_gate(net, ep, init_norms, log)
            if not ok:
                killed = msg
                P(log, f"\n{'!'*64}\n[KILL] {msg}\n"
                       f"[KILL] stopping at ep{ep}; NOT saving any ckpt; routing to Lead/debugger.\n{'!'*64}")
                break

        if ep == 30:
            ck = GDF.make_checkpoint(net, epoch=ep, comment=f"#75 PATH B L8-full bs={BS} seed{SEED}")
            torch.save(ck, CKPT_EP30); ckpts["30"] = CKPT_EP30
            P(log, f"[ep{ep:>2}] saved checkpoint {CKPT_EP30}")
        if ep == N_EPOCHS - 1:
            ck = GDF.make_checkpoint(net, epoch=ep, comment=f"#75 PATH B L8-full bs={BS} seed{SEED}")
            torch.save(ck, CKPT_EP79); ckpts["79"] = CKPT_EP79
            P(log, f"[ep{ep:>2}] saved checkpoint {CKPT_EP79}")

    wall = time.time() - wall0

    # end-of-run vitals
    fin_norms = weight_norms(net)
    fin_bad = all_weights_finite(net)
    fin_rate = msi_rate(net)
    n_done = len(per_ep)
    P(log, f"\n{'='*64}")
    if killed:
        P(log, f"[RESULT] KILLED at ep{n_done-1}: {killed}")
    else:
        P(log, f"[RESULT] COMPLETE — {n_done} epochs")
    P(log, f"[RESULT] MEASURED 80-ep wall = {wall:.2f}s ({wall/60:.3f} min)   "
           f"vs 600s budget -> {'UNDER (<=10 min)' if wall <= 600 else 'OVER'}")
    if n_done:
        P(log, f"[RESULT] per-epoch mean = {wall/n_done:.3f}s   "
               f"(le25 mean={np.mean(per_ep[:26]):.3f}s, gt25 mean={np.mean(per_ep[26:]):.3f}s)"
               if n_done > 26 else f"[RESULT] per-epoch mean = {wall/n_done:.3f}s")
    P(log, f"[RESULT] end-of-run msi_rate={fin_rate:.4e}  finite={'OK' if not fin_bad else 'BAD:'+','.join(fin_bad)}")
    P(log, f"[RESULT] end-of-run weight norms (init->final):")
    for nm in MON_W:
        if nm in fin_norms and nm in init_norms:
            r = fin_norms[nm] / init_norms[nm] if init_norms[nm] > 0 else float("nan")
            P(log, f"           {nm:14s} {init_norms[nm]:.4e} -> {fin_norms[nm]:.4e}  ({r:.3f}x)")
    P(log, f"[RESULT] ckpts: {ckpts}")

    json.dump(dict(wall=wall, per_ep=per_ep, n_epochs_done=n_done, killed=killed,
                   ckpts=ckpts, init_norms=init_norms, final_norms=fin_norms,
                   final_msi_rate=fin_rate, final_finite=(not fin_bad),
                   seed=SEED, bs=BS, n_seq=N_SEQ),
              open(os.path.join(THIS, "retrain_l8full.json"), "w"), indent=1)
    log.close()
    if killed or fin_bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
