#!/usr/bin/env python3
"""TASK #88 — graded-bell screen harness: stimulus-noise + ensemble trial-pooling on the FROZEN
TBW readout.

Re-runs compute_tbw_temporal_fusion_persep's forward (TBW_test.py:900-, build=delayfix) with a
`noise_std` INPUT knob — the existing argument of the production stimulus builder
generate_av_batch_tensor (Training.py), which the readout passes as noise_std=0.0 (TBW_test.py:976).
The fusion CLASSIFIER `TBW.is_temporally_fused` (md5 80d33465...) and its peak/valley finding are
called VERBATIM and byte-frozen — md5 asserted BEFORE and AFTER every run. NOT a readout edit: a
noisier stimulus is presented to the unchanged classifier.

Pooling ("ensemble"): P(fusion)[SOA] = Σ fused / Σ trials across measurement units. A unit =
(ckpt, run_seed); run_seed seeds BOTH the spatial-loc draw (the readout's only RNG, via the #85
seeded_default_rng shim) AND the input noise (torch). SCREEN = seed-42 tau10 × run_seeds {0..4};
FULL ensemble (branch) = seeds 42-46 × run_seeds.

SINGLE-VARIABLE GUARANTEE: the noise_std=0 path of this harness is VERIFIED bit-exact vs the real
V.measure_tbw readout (dmax==0) → noise_std is the ONLY change at noise>0.

Scores vs PREREG_82_bell_criteria.md "TASK #88" (graded-bell GO/NO-GO), LOCKED before this data.
Validator analysis harness — imports val36_traj_d52 (frozen readouts) + harden_85; edits no
production code. cuda:0 only.
"""
import os, sys, json, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import val36_traj_d52 as V          # build=delayfix; load_ckpt/measure_tbw/measure_ei + md5/_edges + T(=Training)
import harden_85 as H               # seeded_default_rng + #82 metric helpers (tbw_metrics/_prom_max/...)

TBW = V.TBW
gen2 = V.T.generate_two_event_offset_seq
genav = V.T.generate_av_batch_tensor

# ---- readout forward constants — VERBATIM from compute_tbw_temporal_fusion_persep (TBW_test.py:900-) ----
OFFS = list(range(-30, 31, 2))                       # macro-steps; ×10 ms = -300..+300, 31 bins
T_STEPS, D = 60, 5
SIGMA, VTHR, MPH, MPS, MTOT = 2.0, 0.4, 0.2, 3, 10.0
STIM_IN = 1.0
NT_DEFAULT = 50
TBW_MD5 = "80d33465c4bf55d6e85b5990acb92da7"
SBW_MD5 = "73b7d13626964d851cc090818b728311"
EI_OFF_BASE = 0.668                                  # delay52 baseline (PREREG_82); GB7/K5 reference

CKPT_SEED42_TAU10 = os.path.join(HERE, "checkpoint",
                                 "ckpt_ep79_seed42_bs250_delay52_tau10_dL3.pt")


@torch.no_grad()
def tbw_fused_counts(net, noise_std, run_seed, nt=NT_DEFAULT):
    """Re-run the readout forward with a noise_std INPUT knob; return per-SOA FUSED COUNTS + trials.
    Line-for-line compute_tbw_temporal_fusion_persep (TBW_test.py:949-) — only noise_std differs.
    run_seed seeds BOTH the spatial-loc draw (seeded_default_rng, the readout's only RNG) and the
    input noise (torch). Classification via the FROZEN TBW.is_temporally_fused (verbatim)."""
    n_off = len(OFFS)
    init_gFF, init_sc = net.g_FFinh, net.step_counter            # readout save (TBW_test:949-950)
    with H.seeded_default_rng(int(run_seed)):
        rng = np.random.default_rng()                            # shim → default_rng(run_seed), as the readout calls it
        all_locs = rng.integers(0, net.space_size, size=(n_off, nt))
    loc_seqs, mod_seqs = [], []
    for k, off in enumerate(OFFS):
        for i in range(nt):
            ls, ms = gen2(loc=int(all_locs[k, i]), T=T_STEPS, D=D, offset=off, space_size=net.space_size)
            loc_seqs.append(ls); mod_seqs.append(ms)
    total = n_off * nt
    torch.manual_seed(int(run_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(run_seed))
    xA, xV, mask = genav(loc_seqs, mod_seqs, [False] * total, n=net.n, space_size=net.space_size,
                         sigma_in=net.sigma_in, noise_std=float(noise_std), device=net.device,
                         max_len=T_STEPS, stimulus_intensity=STIM_IN)
    net.g_FFinh = init_gFF; net.step_counter = init_sc           # readout restore (TBW_test:981-982)
    net.reset_state(total)
    rast = torch.zeros((T_STEPS, total), device=net.device)
    with torch.inference_mode():
        for t in range(T_STEPS):
            ret = net.update_all_layers_batch(xA[:, t], xV[:, t], mask[:, t], return_spike_sum=True)
            rast[t] = ret[-1].sum(dim=1)
    rast_np = rast.cpu().numpy().reshape(T_STEPS, n_off, nt)
    fused = np.zeros(n_off, dtype=np.int64)
    for k in range(n_off):
        fused[k] = int(sum(int(TBW.is_temporally_fused(
            rast_np[:, k, i], sigma=SIGMA, valley_threshold=VTHR, min_peak_height=MPH,
            min_peak_separation=MPS, min_total=MTOT)) for i in range(nt)))
    offs_ms = np.array([o * 10 for o in OFFS], float)
    return offs_ms, fused, np.full(n_off, nt, dtype=np.int64)


def pool(units):
    """units = list of (offs_ms, fused_counts, nt_counts). Pool: P[soa] = Σfused / Σtrials."""
    offs = units[0][0]
    for u in units:
        assert np.array_equal(u[0], offs), "offset grids differ across units"
    tot_f = np.sum([u[1] for u in units], axis=0)
    tot_n = np.sum([u[2] for u in units], axis=0)
    return offs, tot_f / tot_n, tot_f, tot_n


def graded_metrics(offs, pf):
    """#82 metrics (P0/prom_max/FWHM/tail/range/peak via harden_85) + the #88 gradedness layer."""
    offs = np.asarray(offs, float); pf = np.asarray(pf, float)
    fit = V.TBW.fit_psychometric_curve(offs, pf, robust_fit=True)
    lo, hi, w50 = V._edges(offs, pf, 0.5)
    r = dict(offsets_ms=offs.tolist(), pfusion=pf.tolist(), fwhm_ms=float(fit["fwhm"]),
             box_width50_ms=float(w50 or 0.0), peak=float(pf.max()))
    m = H.tbw_metrics(r)                          # P0,P_lobe_max,prom_max(+pos/neg),FWHM,box50,peak,tail,rng,pf
    inter = (pf > 0.2) & (pf < 0.8)
    asc = np.argsort(offs)
    m.update(FWHM=float(fit["fwhm"]),
             N_inter=int(inter.sum()),
             N_inter_pos=int((inter & (offs > 0)).sum()),
             N_inter_neg=int((inter & (offs < 0)).sum()),
             max_step=float(np.max(np.abs(np.diff(pf[asc])))))
    return m


def score(m, ei_sync, ei_off, integrity):
    """Score the pooled-curve metrics vs PREREG_82 TASK #88 (LOCKED)."""
    rel_off = abs(ei_off - EI_OFF_BASE) / EI_OFF_BASE
    GB = dict(
        GB1_core=bool(m["P0"] >= 0.95),
        GB2_both_limbs_graded=bool(m["N_inter_pos"] >= 1 and m["N_inter_neg"] >= 1),
        GB3_finite_slope=bool(m["max_step"] <= 0.50),
        GB4_monotone=bool(m["prom_max"] <= 0.10),
        GB5_fwhm=bool(200 <= m["FWHM"] <= 300),
        GB6_tail_range=bool(m["tail"] <= 0.20 and m["rng"] >= 0.60),
        GB7_ei=bool(0.80 <= ei_sync <= 1.25 and rel_off <= 0.20),
        GB8_integrity=bool(integrity),
    )
    K = dict(
        K1_box_persists=bool(m["N_inter"] <= 1 and m["max_step"] >= 0.70),
        K2_core_collapse=bool(m["P0"] < 0.95 or not (180 <= m["FWHM"] <= 320)),
        K3_lobe_returns=bool(m["prom_max"] >= 0.15),
        K4_degenerate=bool(m["peak"] < 0.95 or m["tail"] > 0.30 or m["rng"] < 0.50),
        K5_ei_regressed=bool(ei_sync < 0.80 or ei_sync > 1.30 or rel_off > 0.20),
        K6_smoke=bool(not integrity),
    )
    borderline = dict(
        asym_grading=bool((m["N_inter_pos"] >= 1) ^ (m["N_inter_neg"] >= 1)),
        partial_soften=bool(0.50 < m["max_step"] <= 0.70 and m["N_inter"] >= 2),
        prom_borderline=bool(0.10 < m["prom_max"] <= 0.15),
        fwhm_borderline=bool((180 <= m["FWHM"] < 200) or (300 < m["FWHM"] <= 320)),
    )
    all_gb = all(GB.values()); any_k = any(K.values()); any_bl = any(borderline.values())
    verdict = "GO" if (all_gb and not any_k) else "NO-GO"
    return dict(GB=GB, K=K, borderline=borderline, all_GB=all_gb, any_K=any_k,
                any_borderline=any_bl, verdict=verdict)


def md5s():
    return (V.md5(os.path.join(V.TBW_DIR, "TBW_test.py")),
            V.md5(os.path.join(V.SBW_DIR, "SBW_test.py")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT_SEED42_TAU10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["verify", "screen"], default="verify")
    ap.add_argument("--noise_std", type=float, default=None, help="#87 biological level; REQUIRED for --mode screen")
    ap.add_argument("--run_seeds", default="0,1,2,3,4")
    ap.add_argument("--nt", type=int, default=NT_DEFAULT)
    ap.add_argument("--out_json", default=os.path.join(HERE, "out", "grade_88_screen.json"))
    ap.add_argument("--log", default=os.path.join(HERE, "logs", "grade_88.log"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    LOG = open(args.log, "w", buffering=1)
    def P(*a):
        print(*a, flush=True); print(*a, file=LOG, flush=True)
    t0 = time.time()

    # ---- md5 BEFORE (frozen-readout integrity gate) ----
    md5_tbw_b, md5_sbw_b = md5s()
    P(f"[88] device={device} BUILD={V.BUILD} netsrc={V.md5(V.NETSRC)} TBW.md5={md5_tbw_b} SBW.md5={md5_sbw_b}")
    assert md5_tbw_b == TBW_MD5, "TBW readout md5 drift BEFORE run!"
    assert md5_sbw_b == SBW_MD5, "SBW readout md5 drift BEFORE run!"

    net, res, epoch, g_rec, comment = V.load_ckpt(args.ckpt, args.seed, args.nt, device)
    tau_gaba, tau_nmda_inh, grec = float(net.tau_gaba), float(net.tau_nmda_inh), float(net.g_rec)
    strict_clean = (not res.missing_keys and not res.unexpected_keys)
    P(f"[88] G0: ckpt={os.path.basename(args.ckpt)} ep={epoch} tau_gaba={tau_gaba} tau_nmda_inh={tau_nmda_inh} "
      f"g_rec={grec} missing={list(res.missing_keys)} unexpected={list(res.unexpected_keys)} comment={comment!r}")
    assert abs(tau_gaba - 10.0) < 1e-9, f"tau_gaba={tau_gaba} != trained 10.0 (G0 FAIL)"
    assert abs(tau_nmda_inh - 21.6) < 1e-9, f"tau_nmda_inh={tau_nmda_inh} != 21.6"
    assert abs(grec - 0.1) < 1e-9, f"g_rec={grec} != 0.1"
    assert strict_clean, "strict-load not clean (G0 FAIL)"
    P("[88] G0 PASS: trained tau_gaba=10.0, tau_nmda_inh=21.6, g_rec=0.1, strict-load clean")

    # ---- E/I FIRST (deterministic; before any inference_mode forward — #83 inference-tensor lesson) ----
    ei = V.measure_ei(net)
    ei_sync, ei_off = float(ei["ei_ratio_sync"]), float(ei["ei_ratio_offsetmean"])
    P(f"[88] EI (measured first): sync={ei_sync:.4f} offsetmean={ei_off:.4f}")

    # ---- VERIFY: noise_std=0 path == real V.measure_tbw readout, bit-exact (single-variable gate) ----
    offs, fused0, nt0 = tbw_fused_counts(net, noise_std=0.0, run_seed=0, nt=args.nt)
    pf_mine = fused0 / nt0
    with H.seeded_default_rng(0):
        r_real = V.measure_tbw(net, n_trials=args.nt)
    pf_real = np.asarray(r_real["pfusion"], float)
    dmax = float(np.max(np.abs(pf_mine - pf_real)))
    P(f"[88] VERIFY replica vs real readout @noise=0 (run_seed=0): max|Δpf|={dmax:.2e} "
      f"({'PASS' if dmax == 0.0 else 'FAIL'})")
    assert dmax == 0.0, "replica diverged from the real readout @noise=0 — cannot trust the noise probe"

    # ---- pooling correctness: Σfused/Σtrials ≡ mean-of-(fused/nt) at equal NT; pooled∈[0,1]; Σtrials=M·nt
    u0 = (offs, fused0, nt0)
    o1, p1, tf1, tn1 = pool([u0])
    assert np.array_equal(tf1, fused0) and int(tn1[0]) == args.nt
    of, pf_mean = offs, (np.asarray([fused0 / nt0], float).mean(axis=0))
    _, p2, _, _ = pool([u0, u0])
    pooled_ok = bool(np.allclose(p2, pf_mean) and p2.min() >= 0.0 and p2.max() <= 1.0)
    P(f"[88] POOL self-test: pool([u])==u {bool(np.array_equal(p1, fused0/nt0))}; "
      f"Σfused/Σtrials==mean-of-P {bool(np.allclose(p2, pf_mean))}; pooled∈[0,1] {bool(p2.min()>=0 and p2.max()<=1)}")
    assert pooled_ok, "pooling self-test failed"

    md5_tbw_a, md5_sbw_a = md5s()
    assert md5_tbw_a == TBW_MD5 and md5_sbw_a == SBW_MD5, "readout md5 drift AFTER run!"
    md5_stable = (md5_tbw_a == TBW_MD5 and md5_sbw_a == SBW_MD5 and md5_tbw_b == TBW_MD5 and md5_sbw_b == SBW_MD5)

    if args.mode == "verify":
        P(f"\n[88] VERIFY MODE complete: noise=0 bit-exact (dmax={dmax:.0e}), pooling correct, "
          f"md5 frozen before+after ({md5_tbw_a}). Harness READY; awaiting #87 noise level for the screen.")
        json.dump(dict(mode="verify", dmax_noise0=dmax, pooled_ok=pooled_ok, md5_stable=md5_stable,
                       md5_TBW=md5_tbw_a, md5_SBW=md5_sbw_a, tau_gaba=tau_gaba, tau_nmda_inh=tau_nmda_inh,
                       g_rec=grec, strict_clean=strict_clean, ei_sync=ei_sync, ei_off=ei_off,
                       offs_ms=offs.tolist(), pf_noise0=pf_mine.tolist()),
                  open(os.path.join(HERE, "out", "grade_88_verify.json"), "w"), indent=1)
        P(f"[88] total {time.time()-t0:.1f}s -> out/grade_88_verify.json")
        LOG.close(); return

    # ---- SCREEN MODE (gated on #87 noise level) ----
    assert args.noise_std is not None, "--noise_std (the #87 biological level) REQUIRED for --mode screen"
    run_seeds = [int(x) for x in args.run_seeds.split(",") if x.strip() != ""]
    P(f"\n=== SCREEN: noise_std={args.noise_std} pooled over run_seeds={run_seeds} (M={len(run_seeds)}, "
      f"nt={args.nt} ⇒ {len(run_seeds)*args.nt} trials/SOA) ===")
    units, per_rs = [], []
    for rs in run_seeds:
        tk = time.time()
        u = tbw_fused_counts(net, args.noise_std, rs, args.nt)
        units.append(u)
        mr = graded_metrics(u[0], u[1] / u[2])
        per_rs.append(dict(run_seed=rs, P0=mr["P0"], N_inter=mr["N_inter"], N_inter_pos=mr["N_inter_pos"],
                           N_inter_neg=mr["N_inter_neg"], max_step=mr["max_step"], prom_max=mr["prom_max"],
                           FWHM=mr["FWHM"], peak=mr["peak"]))
        P(f"  [rs={rs}] {time.time()-tk:.1f}s P0={mr['P0']:.3f} N_inter={mr['N_inter']}"
          f"(+{mr['N_inter_pos']}/-{mr['N_inter_neg']}) max_step={mr['max_step']:.3f} "
          f"prom={mr['prom_max']:.3f} FWHM={mr['FWHM']:.1f} peak={mr['peak']:.3f}")
    offs, pf_pool, tf, tn = pool(units)
    assert np.all(np.isfinite(pf_pool)), "non-finite pooled curve"
    m = graded_metrics(offs, pf_pool)

    md5_tbw_a, md5_sbw_a = md5s()
    md5_stable = (md5_tbw_a == TBW_MD5 and md5_sbw_a == SBW_MD5)
    finite = bool(np.all(np.isfinite(pf_pool)) and np.isfinite(m["FWHM"]))
    integrity = bool(md5_stable and strict_clean and abs(tau_gaba - 10.0) < 1e-9
                     and abs(tau_nmda_inh - 21.6) < 1e-9 and abs(grec - 0.1) < 1e-9
                     and finite and dmax == 0.0 and m["peak"] > 0.0)
    sc = score(m, ei_sync, ei_off, integrity)

    # ---- TBW curve string (-160..+160 window) ----
    idx = {int(o): i for i, o in enumerate(offs)}
    win = [o for o in offs if -160 <= o <= 160]
    curve_str = " ".join(f"{int(o):+d}:{pf_pool[idx[int(o)]]:.2f}" for o in win)

    out = dict(mode="screen", ckpt=args.ckpt, seed=args.seed, epoch=epoch, noise_std=args.noise_std,
               run_seeds=run_seeds, nt=args.nt, trials_per_soa=len(run_seeds) * args.nt,
               tau_gaba=tau_gaba, tau_nmda_inh=tau_nmda_inh, g_rec=grec, strict_clean=strict_clean,
               md5_TBW=md5_tbw_a, md5_SBW=md5_sbw_a, md5_stable=md5_stable, dmax_noise0=dmax,
               ei_sync=ei_sync, ei_off=ei_off, offs_ms=offs.tolist(), pf_pooled=pf_pool.tolist(),
               fused_pooled=tf.tolist(), trials_pooled=tn.tolist(), per_run_seed=per_rs,
               metrics=dict((k, (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v))
                            for k, v in m.items() if k != "pf"),
               score=sc, integrity=integrity)
    json.dump(out, open(args.out_json, "w"), indent=1)

    # ---- scorecard ----
    P("\n" + "=" * 96)
    P(f"TASK #88 GRADED-BELL SCREEN — seed{args.seed} tau10 ep{epoch} @noise_std={args.noise_std} "
      f"(pooled {len(run_seeds)}×{args.nt}={len(run_seeds)*args.nt} trials/SOA) vs PREREG_82 TASK #88")
    P("=" * 96)
    P(f"  TBW curve [-160..+160]: {curve_str}")
    P(f"  {'metric':<16}{'value':>10}    GO band")
    P(f"  {'P0':<16}{m['P0']:>10.3f}    >=0.95")
    P(f"  {'N_inter(0.2,0.8)':<16}{m['N_inter']:>10d}    pos>=1 AND neg>=1  (pos={m['N_inter_pos']} neg={m['N_inter_neg']})")
    P(f"  {'max_step':<16}{m['max_step']:>10.3f}    <=0.50   (KILL>=0.70)")
    P(f"  {'prom_max':<16}{m['prom_max']:>10.3f}    <=0.10   (KILL>=0.15)")
    P(f"  {'P_lobe_max':<16}{m['P_lobe_max']:>10.3f}    (ref; plateau-edge, not a gate)")
    P(f"  {'FWHM(ms)':<16}{m['FWHM']:>10.1f}    [200,300] (KILL outside [180,320])")
    P(f"  {'tail':<16}{m['tail']:>10.3f}    <=0.20")
    P(f"  {'range':<16}{m['rng']:>10.3f}    >=0.60")
    P(f"  {'peak':<16}{m['peak']:>10.3f}    >=0.95")
    P(f"  {'EI_sync':<16}{ei_sync:>10.3f}    [0.80,1.25]")
    P(f"  {'EI_off':<16}{ei_off:>10.3f}    |d|/{EI_OFF_BASE}<=0.20 -> {abs(ei_off-EI_OFF_BASE)/EI_OFF_BASE:.3f}")
    P("  --- GB (GO requires ALL True) ---")
    for k, v in sc["GB"].items():
        P(f"     [{'PASS' if v else 'FAIL'}] {k}")
    P("  --- K (any True => NO-GO) ---")
    for k, v in sc["K"].items():
        P(f"     [{'TRIP' if v else ' ok '}] {k}")
    P("  --- borderline flags ---")
    for k, v in sc["borderline"].items():
        P(f"     [{'FLAG' if v else ' -- '}] {k}")
    P(f"  integrity_ok={integrity}  md5_stable={md5_stable}  finite={finite}  dmax_noise0={dmax:.0e}")
    P(f"  >>> VERDICT: {sc['verdict']}   (all_GB={sc['all_GB']}, any_K={sc['any_K']}, borderline={sc['any_borderline']})")
    P(f"  total {time.time()-t0:.1f}s -> {args.out_json}")
    P("=" * 96)
    LOG.close()


if __name__ == "__main__":
    main()
