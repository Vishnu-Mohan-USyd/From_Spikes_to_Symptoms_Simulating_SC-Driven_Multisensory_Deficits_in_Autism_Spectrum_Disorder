#!/usr/bin/env python3
"""TASK #85 — characterize TBW/SBW persep readout run-to-run non-determinism + harden the #83
bell measurement SAMPLING protocol. READ-ONLY on production: imports val36_traj_d52 (frozen
readouts TBW_test 80d33465 / SBW_test 73b7d136, is_temporally_fused UNTOUCHED). #82 thresholds
UNCHANGED. The only levers explored are n_trials / n_runs averaging / RNG seeding.

SOURCE (proven from code, confirmed here): both persep readouts call np.random.default_rng() with
NO seed (TBW_test:951, SBW_test:472) → fresh OS entropy per call drives the random trial spatial
locations (TBW all_locs:959; SBW base_deg:520). np.random.seed()/torch.manual_seed() cannot control
it (default_rng ignores the legacy global seed). Stimuli are built noise_std=0.0 and the forward is
torch.inference_mode (EI, same forward, is deterministic). Control = patch np.random.default_rng at
runtime (a seeding lever; readout file/md5 UNCHANGED) OR average more samples (n_trials / n_runs).
"""
import os, sys, json, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import val36_traj_d52 as V  # build=delayfix; provides build_net/load_ckpt/measure_tbw/measure_sbw/measure_ei + md5/NETSRC

# ---- runtime seeding shim for the readout's unseeded np.random.default_rng (no readout edit) ----
_ORIG_DEFAULT_RNG = np.random.default_rng
class seeded_default_rng:
    """Patch np.random.default_rng so the readout's `np.random.default_rng()` returns a Generator
    seeded with `seed`. Restores on exit. This is the ONLY external injection point (the readout
    ignores the legacy global seed). Readout source/md5 unchanged."""
    def __init__(self, seed): self.seed = int(seed)
    def __enter__(self):
        np.random.default_rng = (lambda *a, **k: _ORIG_DEFAULT_RNG(self.seed))
        return self
    def __exit__(self, *exc):
        np.random.default_rng = _ORIG_DEFAULT_RNG

# ---- #82 decision metrics on a P(fusion)-vs-SOA curve (verbatim definitions from PREREG_82) ----
def _p0(offs, pf):
    offs = np.asarray(offs, float); pf = np.asarray(pf, float)
    return float(pf[int(np.argmin(np.abs(offs)))])

def _p_lobe_max(offs, pf):
    offs = np.asarray(offs, float); pf = np.asarray(pf, float)
    m = (np.abs(offs) >= 100) & (np.abs(offs) <= 160)
    return float(pf[m].max())

def _prom_max(offs, pf):
    """Window-free running-min re-ascent prominence, per side, walking OUTWARD from SOA 0."""
    offs = np.asarray(offs, float); pf = np.asarray(pf, float)
    asc = np.argsort(offs)                      # indices for offs -300..+300 ascending
    pos = [i for i in asc if offs[i] >= 0]                  # 0,20,...,300
    neg = [i for i in asc[::-1] if offs[i] <= 0]            # 0,-20,...,-300
    def prom(idx):
        seq = pf[idx]; rm = np.minimum.accumulate(seq); return float((seq - rm).max())
    pp, pn = prom(pos), prom(neg)
    return max(pp, pn), pp, pn

def _tail_range(offs, pf):
    offs = np.asarray(offs, float); pf = np.asarray(pf, float)
    tail = float(pf[(np.abs(offs) >= 280) & (np.abs(offs) <= 300)].max())
    return tail, _p0(offs, pf) - tail

def tbw_metrics(r):
    offs, pf = r["offsets_ms"], r["pfusion"]
    prom, pp, pn = _prom_max(offs, pf)
    tail, rng = _tail_range(offs, pf)
    return dict(P0=_p0(offs, pf), P_lobe_max=_p_lobe_max(offs, pf),
                prom_max=prom, prom_pos=pp, prom_neg=pn,
                FWHM=float(r["fwhm_ms"]), box50=float(r["box_width50_ms"]),
                peak=float(r["peak"]), tail=tail, rng=rng, pf=list(map(float, pf)))

def msd(xs):
    a = np.asarray(xs, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0

def summ(label, key, rows):
    m, s = msd([r[key] for r in rows])
    lo, hi = min(r[key] for r in rows), max(r[key] for r in rows)
    return dict(label=label, key=key, mean=m, sd=s, min=lo, max=hi, n=len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoint",
                                                    "ckpt_ep79_seed42_bs250_delay52_tau10_dL3.pt"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=15)          # base repeats at n_trials=50
    ap.add_argument("--ntrials_base", type=int, default=50)
    ap.add_argument("--sweep", default="100,200")          # n_trials sweep (TBW), K_sweep each
    ap.add_argument("--K_sweep", type=int, default=5)
    ap.add_argument("--out_json", default=os.path.join(HERE, "out", "harden_85_seed42.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    t0 = time.time()
    print(f"[85] device={device} BUILD={V.BUILD} netsrc.md5={V.md5(V.NETSRC)} "
          f"TBW.md5={V.md5(os.path.join(V.TBW_DIR,'TBW_test.py'))} "
          f"SBW.md5={V.md5(os.path.join(V.SBW_DIR,'SBW_test.py'))}", flush=True)
    assert V.md5(os.path.join(V.TBW_DIR, "TBW_test.py")) == "80d33465c4bf55d6e85b5990acb92da7", "TBW readout md5 drift!"
    assert V.md5(os.path.join(V.SBW_DIR, "SBW_test.py")) == "73b7d13626964d851cc090818b728311", "SBW readout md5 drift!"

    net, res, epoch, g_rec, comment = V.load_ckpt(args.ckpt, args.seed, 256, device)
    print(f"[85] loaded {os.path.basename(args.ckpt)} ep={epoch} missing={list(res.missing_keys)} "
          f"unexpected={list(res.unexpected_keys)} tau_nmda_inh={float(net.tau_nmda_inh)} "
          f"tau_gaba={float(net.tau_gaba)} g_rec={float(net.g_rec)} plast={net.plasticity_enabled} "
          f"comment={comment!r}", flush=True)
    out = dict(ckpt=args.ckpt, seed=args.seed, epoch=epoch, tau_nmda_inh=float(net.tau_nmda_inh),
               tau_gaba=float(net.tau_gaba), g_rec=float(net.g_rec),
               md5_TBW=V.md5(os.path.join(V.TBW_DIR, "TBW_test.py")),
               md5_SBW=V.md5(os.path.join(V.SBW_DIR, "SBW_test.py")), strict_clean=(not res.missing_keys and not res.unexpected_keys))

    NT = args.ntrials_base
    # ============ STEP 1 — SOURCE: seeded vs unseeded cascade ============
    print("\n=== STEP1 SOURCE (n_trials=%d) ===" % NT, flush=True)
    def pf_of(r): return np.asarray(r["pfusion"], float)
    def diff(a, b): return float(np.max(np.abs(pf_of(a) - pf_of(b)))), int(np.sum(pf_of(a) != pf_of(b)))

    u1 = V.measure_tbw(net, n_trials=NT); u2 = V.measure_tbw(net, n_trials=NT)
    un_max, un_nd = diff(u1, u2)
    print(f"[SRC] UNSEEDED tbw x2: max|Δpf|={un_max:.4f} ndiff={un_nd}/31", flush=True)
    with seeded_default_rng(args.seed):
        s1 = V.measure_tbw(net, n_trials=NT); s2 = V.measure_tbw(net, n_trials=NT)
    se_max, se_nd = diff(s1, s2)
    print(f"[SRC] SEEDED(default_rng={args.seed}) tbw x2: max|Δpf|={se_max:.4f} ndiff={se_nd}/31", flush=True)
    # torch-noise probe: seed default_rng AND torch before each call
    def seeded_both():
        with seeded_default_rng(args.seed):
            torch.manual_seed(args.seed);
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
            ra = V.measure_tbw(net, n_trials=NT)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
            rb = V.measure_tbw(net, n_trials=NT)
        return diff(ra, rb)
    sb_max, sb_nd = seeded_both()
    print(f"[SRC] SEEDED default_rng+torch tbw x2: max|Δpf|={sb_max:.4f} ndiff={sb_nd}/31", flush=True)
    # EI determinism
    e1 = V.measure_ei(net); e2 = V.measure_ei(net)
    ei_d = abs(e1["ei_ratio_sync"] - e2["ei_ratio_sync"]) + abs(e1["ei_ratio_offsetmean"] - e2["ei_ratio_offsetmean"])
    print(f"[SRC] EI x2: |Δsync|+|Δoff|={ei_d:.3e}", flush=True)
    out["source"] = dict(unseeded=dict(max_abs=un_max, ndiff=un_nd),
                         seeded_default_rng=dict(max_abs=se_max, ndiff=se_nd),
                         seeded_default_rng_plus_torch=dict(max_abs=sb_max, ndiff=sb_nd),
                         ei_delta=ei_d,
                         verdict=("RNG default_rng locations (seeding default_rng → reproducible)"
                                  if se_max == 0.0 else
                                  ("torch-noise residual (default_rng+torch → reproducible)" if sb_max == 0.0
                                   else "RESIDUAL after full seeding → GPU atomics/other (ESCALATE)")))
    print(f"[SRC] VERDICT: {out['source']['verdict']}", flush=True)

    # ============ STEP 2/3 — QUANTIFY base K + n_trials sweep (TBW) ============
    print(f"\n=== STEP2 QUANTIFY tbw n_trials={NT} K={args.K} ===", flush=True)
    base = []
    for k in range(args.K):
        tk = time.time(); r = V.measure_tbw(net, n_trials=NT); base.append(tbw_metrics(r))
        print(f"  [base {k+1}/{args.K}] {time.time()-tk:.1f}s P_lobe={base[-1]['P_lobe_max']:.3f} "
              f"prom={base[-1]['prom_max']:.3f} FWHM={base[-1]['FWHM']:.1f} P0={base[-1]['P0']:.3f}", flush=True)
    out["base_curves"] = [b["pf"] for b in base]
    out["base_offsets_ms"] = u1["offsets_ms"]
    metrics = ["P_lobe_max", "prom_max", "FWHM", "box50", "P0", "peak", "tail"]
    out["quantify_base"] = {m: summ(f"tbw_nt{NT}", m, base) for m in metrics}

    sweep = {}
    for nt in [int(x) for x in args.sweep.split(",") if x.strip()]:
        rows = []
        for k in range(args.K_sweep):
            tk = time.time(); r = V.measure_tbw(net, n_trials=nt); rows.append(tbw_metrics(r))
            print(f"  [nt={nt} {k+1}/{args.K_sweep}] {time.time()-tk:.1f}s P_lobe={rows[-1]['P_lobe_max']:.3f} "
                  f"prom={rows[-1]['prom_max']:.3f} FWHM={rows[-1]['FWHM']:.1f}", flush=True)
        sweep[str(nt)] = {m: summ(f"tbw_nt{nt}", m, rows) for m in metrics}
    out["sweep_ntrials"] = sweep

    # ---- n_runs averaging lever, bootstrapped from the K base curves (no extra compute) ----
    print(f"\n=== STEP3 n_runs averaging (bootstrap from {len(base)} base n_trials={NT} curves) ===", flush=True)
    offs = np.asarray(u1["offsets_ms"], float)
    curves = np.asarray([b["pf"] for b in base], float)            # (K, 31)
    rngb = np.random.default_rng(12345)
    nruns = {}
    for M in [1, 2, 3, 4, 6, 8]:
        if M > len(base):
            continue
        pl, pr, fw = [], [], []
        for _ in range(4000):
            pick = rngb.choice(len(base), size=M, replace=False) if M <= len(base) else rngb.choice(len(base), size=M, replace=True)
            mc = curves[pick].mean(axis=0)
            pl.append(_p_lobe_max(offs, mc)); pr.append(_prom_max(offs, mc)[0])
        nruns[str(M)] = dict(P_lobe_max_sd=float(np.std(pl, ddof=1)), P_lobe_max_mean=float(np.mean(pl)),
                             prom_max_sd=float(np.std(pr, ddof=1)), prom_max_mean=float(np.mean(pr)),
                             note="metrics computed on the M-run-AVERAGED curve")
        print(f"  M={M}: P_lobe_max sd={nruns[str(M)]['P_lobe_max_sd']:.4f}  prom_max sd={nruns[str(M)]['prom_max_sd']:.4f}", flush=True)
    out["nruns_avg_bootstrap"] = nruns
    # 5-seed-mean projection (decision in #82 is on the 5-seed mean; SEM = single-seed sd / sqrt(5))
    plb = out["quantify_base"]["P_lobe_max"]["sd"]; prb = out["quantify_base"]["prom_max"]["sd"]; fwb = out["quantify_base"]["FWHM"]["sd"]
    out["five_seed_mean_projection_nt50"] = dict(P_lobe_max_sem=plb/np.sqrt(5), prom_max_sem=prb/np.sqrt(5),
                                                 FWHM_sem=fwb/np.sqrt(5),
                                                 note="if measurement-noise-dominated; real seed-to-seed spread adds on top (see aggregate SDs)")

    # ============ SBW + EI determinism ============
    print(f"\n=== SBW n_trials={NT} K={args.K} (margin 5deg, observed swing ~0.57deg) ===", flush=True)
    sbw = []
    for k in range(args.K):
        tk = time.time(); s = V.measure_sbw(net, n_trials=NT)
        sbw.append(dict(SBW_hw=float(s["halfwidth_deg"]), in_band=bool(s["in_band_24p5_40p9"])))
        print(f"  [sbw {k+1}/{args.K}] {time.time()-tk:.1f}s hw={sbw[-1]['SBW_hw']:.3f} in_band={sbw[-1]['in_band']}", flush=True)
    out["quantify_sbw"] = dict(SBW_hw=summ(f"sbw_nt{NT}", "SBW_hw", sbw),
                               in_band_frac=float(np.mean([1.0 if x["in_band"] else 0.0 for x in sbw])))
    out["ei_determinism"] = dict(delta_sum=ei_d, deterministic=bool(ei_d == 0.0),
                                 ei_ratio_sync=e1["ei_ratio_sync"], ei_ratio_offsetmean=e1["ei_ratio_offsetmean"])

    json.dump(out, open(args.out_json, "w"), indent=1)
    # ---- console summary ----
    print("\n" + "=" * 90)
    print("TASK #85 SUMMARY  (ckpt seed42 tau_gaba=%.1f)" % out["tau_gaba"])
    print(f"  SOURCE: {out['source']['verdict']}")
    print(f"          unseeded max|Δpf|={un_max:.3f} ndiff={un_nd}/31  |  seeded(default_rng) max|Δpf|={se_max:.3f}  |  EI Δ={ei_d:.1e}")
    qb = out["quantify_base"]
    print(f"  n_trials={NT} (single-seed run-to-run, K={args.K}):")
    for m in ["P_lobe_max", "prom_max", "FWHM", "box50", "P0", "SBW_hw" if False else "peak"]:
        print(f"     {m:<11} mean={qb[m]['mean']:.4f} sd={qb[m]['sd']:.4f} range[{qb[m]['min']:.3f},{qb[m]['max']:.3f}]")
    print(f"     SBW_hw      mean={out['quantify_sbw']['SBW_hw']['mean']:.3f} sd={out['quantify_sbw']['SBW_hw']['sd']:.3f}")
    print("  n_runs averaging (bootstrap, metrics on averaged curve):")
    for M, d in out["nruns_avg_bootstrap"].items():
        print(f"     M={M}: P_lobe_max sd={d['P_lobe_max_sd']:.4f}  prom_max sd={d['prom_max_sd']:.4f}")
    print(f"  5-seed-mean projection @nt50: P_lobe SEM={out['five_seed_mean_projection_nt50']['P_lobe_max_sem']:.4f} "
          f"prom SEM={out['five_seed_mean_projection_nt50']['prom_max_sem']:.4f} FWHM SEM={out['five_seed_mean_projection_nt50']['FWHM_sem']:.1f}ms")
    print(f"  total {time.time()-t0:.1f}s -> {args.out_json}")
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
