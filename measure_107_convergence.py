#!/usr/bin/env python3
"""TASK #107 — CONVERGENCE READ: does the baked-in correlated ΔL jitter (sigma_dL=30 ms) still deliver
the GRADED TBW BELL at FULL convergence (ep79), while the no-jitter ep79 ref stays a BOX?

Reuses the #104 develop-check MEASUREMENT PATH verbatim: imports measure_develop_check.py (MDC) and
calls its frozen jitter readout (tbw_fused_counts_jit — line-for-line the #88 readout forward + per-trial
common-mode sigma_dL via the build's gen2 arg; classification via the FROZEN TBW.is_temporally_fused,
md5 80d33465). Only the ckpt/epoch/seed are re-pointed to ep79. Importing MDC is side-effect-safe (its
log is opened inside main(), which we do NOT call).

INDICATIVE single-TRAINING-seed read (one seed × pooled M=5 measurement run-seeds = 250 trials/SOA),
explicitly NOT the final 5-seed pooled GO/NO-GO (that needs the #92 per-seed-fit-then-aggregate gate).

Build seed is provably irrelevant to the loaded net (assign_unimodal_preferred_locations evenly tiles
i/(n-1)·(sp-1), NO rng, and the locs are NOT in the 19-key state_dict; all params strict-loaded) — but
we build each net with its MATCHING training seed anyway (the only seed-leak path is the canonical-Hz
probe RNG). Frozen TBW md5 80d33465 + SBW 73b7d136 asserted BEFORE and AFTER. One ckpt per process.
tau_nmda_inh=21.6, tau_gaba=10.0 forced/asserted on load; net.sigma_dL_frames must be 0 (jitter enters
via the readout arg). cuda per CUDA_VISIBLE_DEVICES. Validator harness — edits no production code.
"""
import os, sys, json, time, argparse
# self-locating paths (flat layout: measure_develop_check is a sibling)
VAL = os.path.dirname(os.path.abspath(__file__))
MEASOPT = VAL  # flat: siblings
sys.path.insert(0, MEASOPT)            # measure_develop_check
sys.path.insert(0, VAL)
import numpy as np
import torch
import measure_develop_check as MDC    # sets TAU_GABA=10 / SIGMA_DL_FRAMES=0.0 / VAL36_BUILD=delayfix, then imports V/H/G88
V, H, G88 = MDC.V, MDC.H, MDC.G88

NT = MDC.NT                             # 50
RUN_SEEDS = MDC.RUN_SEEDS               # [0,1,2,3,4] -> 250 trials/SOA pooled
TBW_MD5, SBW_MD5, EI_OFF_BASE = G88.TBW_MD5, G88.SBW_MD5, G88.EI_OFF_BASE


def canon_hz(ckpt, seed, device, tag, P):
    """EVAL-path canonical MSI Hz (verbatim MDC.canonical_msi_hz, but matching build seed)."""
    net, res, epoch, g_rec, comment = V.load_ckpt(ckpt, seed, NT, device)
    strict = (not res.missing_keys and not res.unexpected_keys)
    P(f"  [{tag}] load ep={epoch} tau_gaba={float(net.tau_gaba)} tau_nmda_inh={float(net.tau_nmda_inh)} "
      f"g_rec={float(net.g_rec)} net.sigma_dL_frames={float(net.sigma_dL_frames)} strict_clean={strict} "
      f"missing={list(res.missing_keys)} unexpected={list(res.unexpected_keys)}")
    assert strict, f"{tag}: ckpt did not load strict-clean"
    m = V.T.run_sc_diagnostics(net, modality="B", verbose=False)
    rates = {k: float(v) for k, v in m["spike_rates"].items()}
    ie = float(m.get("I_E_ratio", float("nan")))
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rates.get("MSI", float("nan")), rates, ie, epoch


def measure(ckpt, seed, sigma_dL, tag, label, device, P):
    t0 = time.time()
    md5_tbw_b, md5_sbw_b = MDC.md5s()
    P(f"[107 {label}] BUILD={V.BUILD} netsrc={V.md5(V.NETSRC)} TBW={md5_tbw_b} SBW={md5_sbw_b}")
    assert md5_tbw_b == TBW_MD5 and md5_sbw_b == SBW_MD5, "readout md5 drift BEFORE run!"

    # ---- canonical MSI Hz (EVAL path) ----
    P(f"\n=== [{label}] canonical MSI Hz (run_sc_diagnostics bimodal probe; real Hz) ===")
    msi_hz, rates, ie, ep_h = canon_hz(ckpt, seed, device, tag, P)
    P(f"  {tag}: MSI={msi_hz:.2f} Hz  I/E={ie:.3f}  rates={ {k: round(v,2) for k,v in rates.items()} }")

    # ---- measurement net (matching build seed) ----
    net, res, epoch, g_rec, comment = V.load_ckpt(ckpt, seed, NT, device)
    strict = (not res.missing_keys and not res.unexpected_keys)
    tau_gaba, tni, grec = float(net.tau_gaba), float(net.tau_nmda_inh), float(net.g_rec)
    sdl_net = float(getattr(net, "sigma_dL_frames", -1.0))
    P(f"[107 {label}] G0: ep={epoch} tau_gaba={tau_gaba} tau_nmda_inh={tni} g_rec={grec} "
      f"net.sigma_dL_frames={sdl_net} strict={strict} comment={comment!r}")
    assert strict, "ckpt did not load strict-clean"
    assert abs(tau_gaba - 10.0) < 1e-9 and abs(tni - 21.6) < 1e-9, "tau mismatch vs reference"
    assert sdl_net == 0.0, "measurement net.sigma_dL_frames must be 0 (jitter enters via the readout arg)"
    assert epoch == 79, f"ckpt epoch {epoch} != 79"

    # ---- EI FIRST (in-place reset must precede any inference_mode forward, #83) ----
    ei = V.measure_ei(net)
    ei_sync, ei_off = float(ei["ei_ratio_sync"]), float(ei["ei_ratio_offsetmean"])
    rel_off = abs(ei_off - EI_OFF_BASE) / EI_OFF_BASE
    P(f"[107 {label}] EI: sync={ei_sync:.4f} offmean={ei_off:.4f} rel_off={rel_off:.3f}")

    # ---- TBW with jitter (INTRINSIC), pooled M=5 ----
    P(f"\n=== [{label}] TBW @sigma_dL={sigma_dL*10:.0f}ms pooled run_seeds={RUN_SEEDS} "
      f"({len(RUN_SEEDS)*NT} trials/SOA) ===")
    units, per_rs, all_eff = [], [], []
    for rs in RUN_SEEDS:
        tk = time.time()
        o, f, n, eff = MDC.tbw_fused_counts_jit(net, sigma_dL, run_seed=rs, nt=NT)
        units.append((o, f, n)); all_eff.append(eff)
        mr = G88.graded_metrics(o, f / n)
        per_rs.append(dict(run_seed=rs, P0=float(mr["P0"]), N_inter=int(mr["N_inter"]),
                           N_inter_pos=int(mr["N_inter_pos"]), N_inter_neg=int(mr["N_inter_neg"]),
                           max_step=float(mr["max_step"]), prom_max=float(mr["prom_max"]), FWHM=float(mr["FWHM"])))
        P(f"  [rs={rs}] {time.time()-tk:.1f}s P0={mr['P0']:.3f} N_inter={mr['N_inter']}"
          f"(+{mr['N_inter_pos']}/-{mr['N_inter_neg']}) max_step={mr['max_step']:.3f} "
          f"prom={mr['prom_max']:.3f} FWHM={mr['FWHM']:.1f}")
    offs, pf_pool, tf, tn = G88.pool(units)
    m = G88.graded_metrics(offs, pf_pool)
    bm = MDC.band_metrics(offs, pf_pool)
    eff_sd_ms = float(np.concatenate(all_eff).std(ddof=1) * 10.0)

    # ---- SBW LAST (frozen readout; sigma_dL does not enter it -> ckpt baseline) ----
    with H.seeded_default_rng(0):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        sbw = V.measure_sbw(net, n_trials=NT)
    sbw_hw, sbw_in = float(sbw["halfwidth_deg"]), bool(sbw["in_band_24p5_40p9"])
    P(f"[107 {label}] SBW halfwidth={sbw_hw:.2f} deg in_band[24.5,40.9]={sbw_in}")

    idx = {int(round(o)): i for i, o in enumerate(offs)}
    def pf_at(ms):
        return float(pf_pool[idx[ms]]) if ms in idx else float("nan")
    fus = {"-200": pf_at(-200), "0": pf_at(0), "+200": pf_at(200)}

    # ---- integrity + score ----
    md5_tbw_a, md5_sbw_a = MDC.md5s()
    md5_stable = (md5_tbw_a == TBW_MD5 and md5_sbw_a == SBW_MD5)
    finite = bool(np.all(np.isfinite(pf_pool)) and np.isfinite(m["FWHM"]))
    integrity = bool(md5_stable and strict and abs(tau_gaba - 10.0) < 1e-9
                     and abs(tni - 21.6) < 1e-9 and abs(grec - 0.1) < 1e-9 and finite and m["peak"] > 0.0)
    sc = G88.score(m, ei_sync, ei_off, integrity)
    K1 = bool(sc["K"]["K1_box_persists"])
    max_step = float(m["max_step"])
    is_box = bool(max_step >= 0.70 or K1)             # the develop-check box test

    curve_str = " ".join(f"{int(round(o)):+d}:{pf_pool[i]:.2f}" for i, o in enumerate(offs) if -160 <= o <= 160)

    out = dict(task=107, label=label, tag=tag, ckpt=ckpt, seed=seed, sigma_dL_frames=sigma_dL,
               epoch=epoch, run_seeds=RUN_SEEDS, nt=NT, trials_per_soa=len(RUN_SEEDS) * NT,
               tau_gaba=tau_gaba, tau_nmda_inh=tni, g_rec=grec, net_sigma_dL_frames=sdl_net, strict_clean=strict,
               msi_hz=msi_hz, msi_IE=ie, msi_rates=rates,
               ei_sync=ei_sync, ei_off=ei_off, ei_rel_off=rel_off,
               sbw_halfwidth_deg=sbw_hw, sbw_in_band=sbw_in,
               eff_offset_sd_ms=eff_sd_ms, fusion_at_soa=fus,
               offs_ms=offs.tolist(), pf_pooled=pf_pool.tolist(), fused_pooled=tf.tolist(),
               trials_pooled=tn.tolist(), per_run_seed=per_rs, band=bm,
               metrics={k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                        for k, v in m.items() if k != "pf"},
               score=sc, integrity=integrity, md5_stable=md5_stable, K1_box_persists=K1,
               max_step=max_step, is_box=is_box, md5_TBW_after=md5_tbw_a, md5_SBW_after=md5_sbw_a,
               wall_s=time.time() - t0)
    out_json = os.path.join(VAL, "out", f"measure_107_{label}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(out, open(out_json, "w"), indent=1)

    # ---- scorecard ----
    P("\n" + "=" * 100)
    P(f"#107 CONVERGENCE READ — {tag} ep{epoch} @sigma_dL={sigma_dL*10:.0f}ms (pooled {len(RUN_SEEDS)}x{NT}/SOA)")
    P("=" * 100)
    P(f"  eff-offset SD (jitter live): {eff_sd_ms:.1f} ms  (target {sigma_dL*10:.0f})")
    P(f"  TBW curve [-160..+160]: {curve_str}")
    P(f"      {'metric':<18}{'value':>10}    band")
    P(f"      {'P0':<18}{m['P0']:>10.3f}    >=0.95")
    P(f"      {'N_inter(0.2,0.8)':<18}{m['N_inter']:>10d}    pos>=1 & neg>=1 (pos={m['N_inter_pos']} neg={m['N_inter_neg']})")
    P(f"      {'N_inter(0.1,0.9)':<18}{bm['N_inter_0109']:>10d}    >=4 graded-shorthand")
    P(f"      {'max_step':<18}{max_step:>10.3f}    <=0.50 GO ; >=0.70 BOX/KILL")
    P(f"      {'prom_max':<18}{m['prom_max']:>10.3f}    <=0.10")
    P(f"      {'FWHM(ms)':<18}{m['FWHM']:>10.1f}    [200,300] GO (KILL outside [180,320])")
    P(f"      {'tail':<18}{m['tail']:>10.3f}    <=0.20")
    P(f"      {'range':<18}{m['rng']:>10.3f}    >=0.60")
    P(f"      {'peak':<18}{m['peak']:>10.3f}    >=0.95")
    P(f"  EI sync={ei_sync:.3f} [0.80,1.25] off={ei_off:.3f} rel={rel_off:.3f} [<=0.20]")
    P(f"  SBW hw={sbw_hw:.2f} deg in_band[24.5,40.9]={sbw_in}")
    P(f"  fusion@SOA -200ms={fus['-200']:.3f} 0ms={fus['0']:.3f} +200ms={fus['+200']:.3f}")
    P(f"  canonical MSI={msi_hz:.2f} Hz  I/E={ie:.3f}")
    P("  --- GB (GO requires ALL) / K (any => NO-GO) ---")
    for k, v in sc["GB"].items():
        P(f"     GB [{'PASS' if v else 'FAIL'}] {k}")
    for k, v in sc["K"].items():
        P(f"     K  [{'TRIP' if v else ' ok '}] {k}")
    P(f"  >>> verdict={sc['verdict']}  is_box(max_step>=0.70 OR K1)={is_box}  max_step={max_step:.3f}  K1={K1}")
    md5_tbw_f, md5_sbw_f = MDC.md5s()
    assert md5_tbw_f == TBW_MD5 and md5_sbw_f == SBW_MD5, "readout md5 drift AFTER run!"
    P(f"  integrity={integrity} md5_stable={md5_stable} frozen-after TBW={md5_tbw_f}")
    P(f"  total {time.time()-t0:.1f}s -> {out_json}")
    P("=" * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sigma_dL", type=float, required=True, help="frames; 3.0=30ms jitter, 0.0=no-jitter")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--label", required=True, help="short filename-safe id")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(VAL, "logs"), exist_ok=True)
    LOG = open(os.path.join(VAL, "logs", f"measure_107_{args.label}.log"), "w", buffering=1)
    def P(*a):
        print(*a, flush=True); print(*a, file=LOG, flush=True)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    P(f"[107 {args.label}] device={dev}")
    assert os.path.exists(args.ckpt), f"ckpt missing: {args.ckpt}"
    measure(args.ckpt, args.seed, args.sigma_dL, args.tag, args.label, device, P)
    LOG.close()


if __name__ == "__main__":
    main()
