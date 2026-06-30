#!/usr/bin/env python3
"""TEST 1 — Inverse effectiveness on the route-C dL3-ep79 network (VALIDATION_SPEC §TEST 1).

Biology: multisensory enhancement is greatest when unisensory drive is weakest and declines as
stimulus intensity rises (negative MEI-vs-intensity slope). Manuscript context: MEI 1.1 @0.05 →
0.78 @1.6 a.u.

Method (distilled VERBATIM from inverse_effectiveness_test.py:624 `integrated_spikes` + main):
  per intensity I ∈ {0.05,0.1,0.2,0.4,0.8,1.6} and condition A/V/B:
    Gaussian pulse @ centre_deg=90, width sigma_in, scaled by I, on frames 0..PULSE_LEN-1 of N_FRAMES;
    reset_state(B); per frame `*_, sSum = update_all_layers_batch(xA,xV, return_spike_sum=True)`;
    pop = Σ_frames Σ_neurons sSum.   R_cond,I = mean over n_trials.
  MEI(I) = (R_B − max(R_A,R_V)) / max(R_A,R_V).  Pool MEI across the 5 seeds (validator).

Spec divergences honoured (§1g): drive with net.sigma_in (=10.0, the trained width), NOT master's
5.0 — recorded in JSON. Faithful as-built net (no neuron-param overrides).

Firewall: net.eval(), plasticity off, single torch.no_grad(); per-seed weight bit-identity asserted
before==after; stage_flat TBW/SBW md5 asserted before+after. Local GPU only.

Usage:
  python inverse_effectiveness_routec.py --seed 42 --device cuda:0     # one seed -> per-seed JSON + scorecard
  python inverse_effectiveness_routec.py --aggregate                   # pool seeds -> aggregate JSON + SVG
"""
import os, sys, json, argparse, time, glob
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import routec_net_io as N

METRIC = "inverse_effectiveness"
INTENSITIES = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
CONDS = ["A", "V", "B"]
LOC_DEG = 90.0
PULSE_LEN = 10            # external frames pulse is on (~100 ms)
N_FRAMES = 20            # + quiet tail
DEFAULT_NTRIALS = 24     # within-seed trials (master used 1/model over 10 models; we pool 5 seeds)


@torch.no_grad()
def measure_ie(net, sigma_in, n_trials, device):
    """Return per-(cond,intensity) integrated MSI spike sums and MEI(I). Fully vectorised over cells."""
    n = net.n
    idx_c = LOC_DEG * (n - 1) / (net.space_size - 1)
    xs = torch.arange(n, device=device, dtype=torch.float32)
    cells = [(c, I) for c in CONDS for I in INTENSITIES]            # 18 cells, cond-major
    B = len(cells) * n_trials
    xA = torch.zeros(N_FRAMES, B, n, device=device)
    xV = torch.zeros(N_FRAMES, B, n, device=device)
    for ci, (c, I) in enumerate(cells):
        g = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * I    # (n,)
        sl = slice(ci * n_trials, (ci + 1) * n_trials)
        if c in ("A", "B"):
            xA[:PULSE_LEN, sl, :] = g
        if c in ("V", "B"):
            xV[:PULSE_LEN, sl, :] = g
    net.reset_state(batch_size=B)
    acc = torch.zeros(B, device=device)
    for t in range(N_FRAMES):
        *_, sSum = net.update_all_layers_batch(xA[t], xV[t], return_spike_sum=True)
        acc += sSum.sum(dim=1)                                      # Σ neurons, this frame
    R = acc.reshape(len(cells), n_trials).mean(dim=1).reshape(len(CONDS), len(INTENSITIES))
    R_A = R[0].cpu().numpy(); R_V = R[1].cpu().numpy(); R_B = R[2].cpu().numpy()
    max_uni = np.maximum(R_A, R_V)
    mei = (R_B - max_uni) / np.where(max_uni > 0, max_uni, np.nan)
    return dict(intensities=INTENSITIES, R_A=R_A.tolist(), R_V=R_V.tolist(), R_B=R_B.tolist(),
                max_uni=max_uni.tolist(), mei=mei.tolist())


def run_seed(args):
    device = args.device
    seed = args.seed
    ckpt = args.ckpt or N.ckpt_path_for_seed(seed)
    out_json = args.out_json or os.path.join(N.OUT_DIR, f"{METRIC}_seed{seed}.json")
    os.makedirs(N.OUT_DIR, exist_ok=True)
    t0 = time.time()
    fw_before = N.assert_frozen_readouts("BEFORE")
    net, ld, epoch, g_rec, comment = N.load_ckpt(ckpt, seed, 1, device)
    sigma_in = float(net.sigma_in)
    print(f"[{METRIC} s{seed}] dev={device} epoch={epoch} g_rec={g_rec} tau_nmda_inh={float(net.tau_nmda_inh)} "
          f"tau_gaba={float(net.tau_gaba)} sigma_in={sigma_in} missing={list(ld.missing_keys)} "
          f"unexpected={list(ld.unexpected_keys)} plast={net.plasticity_enabled}", flush=True)
    snap = N.snapshot_weights(net)
    fp_before = N.weight_fingerprint(net)
    with torch.no_grad():
        m = measure_ie(net, sigma_in, args.n_trials, device)
    fp_after = N.assert_weights_unchanged(net, snap, "AFTER")
    fw_after = N.assert_frozen_readouts("AFTER")
    assert fw_after == fw_before
    rec = dict(metric=METRIC, seed=seed, ckpt=os.path.basename(ckpt), epoch=epoch, g_rec=g_rec,
               device=device, build=N.BUILD, tau_nmda_inh=float(net.tau_nmda_inh),
               tau_gaba=float(net.tau_gaba), sigma_in=sigma_in, n_trials=args.n_trials,
               loc_deg=LOC_DEG, pulse_len=PULSE_LEN, n_frames=N_FRAMES,
               plasticity_enabled=bool(net.plasticity_enabled),
               missing_keys=list(ld.missing_keys), unexpected_keys=list(ld.unexpected_keys),
               weight_fp_before=fp_before, weight_fp_after=fp_after,
               weight_bit_identical=bool(fp_before == fp_after),
               md5_TBW=fw_after[0], md5_SBW=fw_after[1], **m)
    json.dump(rec, open(out_json, "w"), indent=1)
    # ---- per-seed scorecard ----
    mei = np.array(m["mei"], float)
    print(f"\n[{METRIC} s{seed}] SCORECARD  ({time.time()-t0:.1f}s)")
    print("  intensity :  " + " ".join(f"{I:>7.2f}" for I in INTENSITIES))
    print("  R_A       :  " + " ".join(f"{v:>7.1f}" for v in m["R_A"]))
    print("  R_V       :  " + " ".join(f"{v:>7.1f}" for v in m["R_V"]))
    print("  R_B(AV)   :  " + " ".join(f"{v:>7.1f}" for v in m["R_B"]))
    print("  MEI       :  " + " ".join(f"{v:>7.3f}" for v in mei))
    d_lo_hi = mei[0] - mei[-1]
    print(f"  MEI(0.05)={mei[0]:.3f}  MEI(1.6)={mei[-1]:.3f}  drop={d_lo_hi:+.3f}  "
          f"monotone_noninc={bool(np.all(np.diff(mei) <= 1e-9))}")
    print(f"  -> wrote {out_json}", flush=True)
    return rec


def _msem(stack):
    stack = np.asarray(stack, float)
    m = np.nanmean(stack, 0)
    sd = np.nanstd(stack, 0, ddof=1) if stack.shape[0] > 1 else np.zeros(stack.shape[1])
    sem = sd / np.sqrt(stack.shape[0])
    return m, sem


def aggregate():
    files = sorted(glob.glob(os.path.join(N.OUT_DIR, f"{METRIC}_seed*.json")))
    assert files, f"no {METRIC}_seed*.json in {N.OUT_DIR}"
    recs = [json.load(open(f)) for f in files]
    seeds = [r["seed"] for r in recs]
    mei = np.array([r["mei"] for r in recs], float)
    mei_m, mei_sem = _msem(mei)
    inten = np.array(INTENSITIES, float)
    # direction summary
    drop = mei[:, 0] - mei[:, -1]
    agg = dict(metric=METRIC, seeds=seeds, n_seeds=len(seeds), intensities=INTENSITIES,
               mei_mean=mei_m.tolist(), mei_sem=mei_sem.tolist(),
               mei_lo_mean=float(mei_m[0]), mei_lo_sem=float(mei_sem[0]),
               mei_hi_mean=float(mei_m[-1]), mei_hi_sem=float(mei_sem[-1]),
               drop_mean=float(np.mean(drop)), drop_sem=float(np.std(drop, ddof=1) / np.sqrt(len(drop))),
               monotone_noninc_mean=bool(np.all(np.diff(mei_m) <= 1e-9)),
               sigma_in=recs[0]["sigma_in"], paper_context="MEI 1.1@0.05 -> 0.78@1.6 (direction is the bar)")
    out_json = os.path.join(N.OUT_DIR, f"{METRIC}_aggregate.json")
    json.dump(agg, open(out_json, "w"), indent=1)
    # ---- figure (house style) ----
    plt = N.setup_house_style(base_fontsize=15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(inten, mei_m, yerr=mei_sem, marker="o", ms=7, capsize=3, lw=2, color="C0",
                label=f"MEI (mean±SEM, n={len(seeds)})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set(xlabel="Stimulus intensity (a.u., log)", ylabel="MEI  (R$_{AV}$−max(R$_A$,R$_V$))/max",
           title="Inverse effectiveness — route-C dL3 ep79")
    ax.legend(frameon=False, loc="best")
    N.despine(ax)
    plt.tight_layout()
    svg = os.path.join(N.OUT_DIR, f"{METRIC}.svg")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(svg.replace(".svg", ".png"), dpi=160, bbox_inches="tight")
    # ---- pooled scorecard ----
    print(f"\n================ {METRIC} AGGREGATE (n={len(seeds)} seeds {seeds}) ================")
    print("  intensity :  " + " ".join(f"{I:>7.2f}" for I in INTENSITIES))
    print("  MEI mean  :  " + " ".join(f"{v:>7.3f}" for v in mei_m))
    print("  MEI sem   :  " + " ".join(f"{v:>7.3f}" for v in mei_sem))
    print(f"  MEI(0.05)={mei_m[0]:.3f}±{mei_sem[0]:.3f}  MEI(1.6)={mei_m[-1]:.3f}±{mei_sem[-1]:.3f}  "
          f"drop={agg['drop_mean']:+.3f}±{agg['drop_sem']:.3f}  monotone_noninc={agg['monotone_noninc_mean']}")
    print(f"  -> {out_json}\n  -> {svg}", flush=True)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_trials", type=int, default=DEFAULT_NTRIALS)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        assert args.seed is not None, "need --seed NN (or --aggregate)"
        run_seed(args)


if __name__ == "__main__":
    main()
