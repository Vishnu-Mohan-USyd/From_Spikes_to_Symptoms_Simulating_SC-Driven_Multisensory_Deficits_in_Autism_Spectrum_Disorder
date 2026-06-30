#!/usr/bin/env python3
"""task #104 POST-HOC — develop-check ep30/ep50 INTRINSIC measurement (single-seed jitter-RETRAIN ckpts).

Runs, on the jitter-trained develop-check ckpts, the gated kill battery the Lead signed off:
  ep50 : TBW-with-jitter (KILL if max_step>=0.70 OR K1_box_persists) + FWHM(WATCH) + SBW + EI
  ep30 : canonical MSI rate (EVAL path = run_sc_diagnostics, real Hz) + fusion@SOA 0/±200 ms + EI
The common-mode latency jitter sigma_dL = 3 frames = 30 ms is injected AT THE READOUT via the build's
gen2 arg, exactly as the frozen #99 screen does. This is the INTRINSIC measurement: the WEIGHTS were
trained with the SAME sigma (develop-check), so jitter is present in both training and measurement.

Single-variable / no-tuning: the readout md5 80d33465(TBW)/73b7d136(SBW) is asserted BEFORE and AFTER.
The 3 jitter-readout fns (eff_offset_from_modseq / tbw_fused_counts_jit / band_metrics) are COPIED
VERBATIM from the frozen screen99_corr_timing_noise.py (same G88/V/H primitives) — copied, not imported,
so importing does not truncate screen99's run log. INFERENCE only; no develop-check ckpt is mutated.
cuda:0 (5090) or cuda:1 (A6000) per CUDA_VISIBLE_DEVICES.  UNVERIFIED until run.

usage:  python measure_develop_check.py {30|50}
"""
import os, sys, json, time, argparse
os.environ.setdefault("TAU_GABA", "10")            # develop-check lineage (V.load_ckpt asserts env==ckpt)
os.environ.setdefault("SIGMA_DL_FRAMES", "0.0")    # MEASUREMENT build: jitter enters via the readout arg, NOT net.sigma_dL_frames
os.environ.setdefault("VAL36_BUILD", "delayfix")   # eager measurement build (matches the delay52 ckpt forward)
import numpy as np
import torch

# self-locating: flat layout - val36_traj_d52 is a sibling; checkpoints live in checkpoint/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VAL = _THIS_DIR  # flat
sys.path.insert(0, VAL)
import val36_traj_d52 as V
import harden_85 as H
import grade_88_noise as G88

gen2 = V.T.generate_two_event_offset_seq           # the EDITED build's sigma_dL-aware SOA generator
genav = V.T.generate_av_batch_tensor
TBW = V.TBW
TBW_MD5, SBW_MD5, EI_OFF_BASE = G88.TBW_MD5, G88.SBW_MD5, G88.EI_OFF_BASE
OFFS = G88.OFFS                                     # macro-steps -30..30 step 2 (x10 ms); 31 SOAs
T_STEPS, D = G88.T_STEPS, G88.D
SIGMA, VTHR, MPH, MPS, MTOT = G88.SIGMA, G88.VTHR, G88.MPH, G88.MPS, G88.MTOT
STIM_IN = G88.STIM_IN

SEED = 42
NT = 50
JIT_RNG_TAG = 99                                   # makes the jitter stream independent of the loc stream
SIGMA_DL_FRAMES_PRIMARY = 3.0                      # 30 ms (FIXED from biology; NOT tuned)
RUN_SEEDS = [0, 1, 2, 3, 4]                         # M=5 pooled units -> 250 trials/SOA (mirrors #88)

CKPT_DIR = os.path.join(_THIS_DIR, "checkpoint")
DEV_CKPT = {30: os.path.join(CKPT_DIR, "ckpt_ep30_seed42_bs250_delay52_tau10_dL3.pt"),
            50: os.path.join(CKPT_DIR, "ckpt_ep50_seed42_bs250_delay52_tau10_dL3.pt")}
BASELINE_EP30 = os.path.join(CKPT_DIR, "ckpt_ep30_seed42_bs250_delay52_tau10.pt")   # knob=0 (no-jitter) baseline

os.makedirs(os.path.join(CKPT_DIR, "out"), exist_ok=True)
LOGP = os.path.join(CKPT_DIR, "out", "measure_develop_check_ep{ep}.log")


def md5s():
    return (V.md5(os.path.join(V.TBW_DIR, "TBW_test.py")), V.md5(os.path.join(V.SBW_DIR, "SBW_test.py")))


# ============================ VERBATIM from screen99_corr_timing_noise.py ============================
def eff_offset_from_modseq(ms):
    """Recover the EFFECTIVE A-V onset offset actually built into a sequence:
    (first index modality in {V,B}) - (first index modality in {A,B}) == eff_offset frames."""
    a0 = next((t for t, c in enumerate(ms) if c in ('A', 'B')), None)
    v0 = next((t for t, c in enumerate(ms) if c in ('V', 'B')), None)
    return (v0 - a0) if (a0 is not None and v0 is not None) else 0


@torch.no_grad()
def tbw_fused_counts_jit(net, sigma_dL_frames, run_seed, nt=NT):
    """#88 readout forward (line-for-line grade_88_noise.tbw_fused_counts) PLUS per-trial common-mode
    sigma_dL jitter injected via the build's gen2 NEW arg. One gen2 call per (SOA,trial) => one shared
    dL per trial. The jitter stream is seeded independently of the spatial-loc stream and reproducibly
    per run_seed. At sigma_dL_frames=0 this is bit-identical to grade_88_noise.tbw_fused_counts(0)."""
    n_off = len(OFFS)
    init_gFF, init_sc = net.g_FFinh, net.step_counter            # readout save (TBW_test:949-950)
    with H.seeded_default_rng(int(run_seed)):
        rng = np.random.default_rng()                            # shim -> default_rng(run_seed): the readout's loc RNG
        all_locs = rng.integers(0, net.space_size, size=(n_off, nt))
    jit_rng = np.random.default_rng([int(run_seed), JIT_RNG_TAG])  # independent of loc stream; reproducible per run_seed
    loc_seqs, mod_seqs, eff = [], [], []
    for k, off in enumerate(OFFS):
        for i in range(nt):
            ls, ms = gen2(loc=int(all_locs[k, i]), T=T_STEPS, D=D, offset=off, space_size=net.space_size,
                          sigma_dL_frames=float(sigma_dL_frames), rng=jit_rng)
            loc_seqs.append(ls); mod_seqs.append(ms); eff.append(eff_offset_from_modseq(ms))
    total = n_off * nt
    torch.manual_seed(int(run_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(run_seed))
    xA, xV, mask = genav(loc_seqs, mod_seqs, [False] * total, n=net.n, space_size=net.space_size,
                         sigma_in=net.sigma_in, noise_std=0.0, device=net.device,
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
    return offs_ms, fused, np.full(n_off, nt, dtype=np.int64), np.array(eff, float)


def band_metrics(offs_ms, pf):
    """N_inter at the two thresholds + max adjacent step, on a -160..+160 window curve."""
    offs_ms = np.asarray(offs_ms, float); pf = np.asarray(pf, float)
    b = (offs_ms >= -160) & (offs_ms <= 160)
    o, p = offs_ms[b], pf[b]
    a = np.argsort(o)
    return dict(N_inter_0208=int(np.sum((p > 0.2) & (p < 0.8))),
                N_inter_0109=int(np.sum((p > 0.1) & (p < 0.9))),
                max_step=float(np.max(np.abs(np.diff(p[a])))) if len(p) > 1 else 0.0)
# ====================================================================================================


def canonical_msi_hz(ckpt_path, device, tag, P):
    """EVAL-path canonical MSI firing rate (Hz): load ckpt fresh, run the build's run_sc_diagnostics
    bimodal Gaussian probe -> spike_rates['MSI'] = _dbg_spk_MSI/(N*sim_time_s) (real Hz, eager so the
    .item() host-syncs execute; NOT the graph-skipped NaN). Returns (msi_hz, all_rates, epoch)."""
    net, res, epoch, g_rec, comment = V.load_ckpt(ckpt_path, SEED, NT, device)
    strict = (not res.missing_keys and not res.unexpected_keys)
    P(f"  [{tag}] load ep={epoch} tau_gaba={float(net.tau_gaba)} tau_nmda_inh={float(net.tau_nmda_inh)} "
      f"g_rec={float(net.g_rec)} sigma_dL_frames={float(net.sigma_dL_frames)} strict_clean={strict} "
      f"missing={list(res.missing_keys)} unexpected={list(res.unexpected_keys)}")
    assert strict, f"{tag}: ckpt did not load strict-clean"
    m = V.T.run_sc_diagnostics(net, modality="B", verbose=False)
    rates = {k: float(v) for k, v in m["spike_rates"].items()}
    msi_hz = rates.get("MSI", float("nan"))
    ie = float(m.get("I_E_ratio", float("nan")))
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return msi_hz, rates, ie, epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("epoch", type=int, choices=[30, 50])
    args = ap.parse_args()
    EP = args.epoch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG = open(LOGP.format(ep=EP), "w", buffering=1)

    def P(*a):
        print(*a, flush=True)
        print(*a, file=LOG, flush=True)

    t0 = time.time()
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    md5_tbw_b, md5_sbw_b = md5s()
    P(f"[dev-check ep{EP}] device={dev} BUILD={V.BUILD} netsrc={V.md5(V.NETSRC)}")
    P(f"[dev-check ep{EP}] frozen readouts BEFORE: TBW={md5_tbw_b} SBW={md5_sbw_b}")
    assert md5_tbw_b == TBW_MD5, "TBW readout md5 drift BEFORE run!"
    assert md5_sbw_b == SBW_MD5, "SBW readout md5 drift BEFORE run!"
    ckpt = DEV_CKPT[EP]
    assert os.path.exists(ckpt), f"develop-check ckpt missing: {ckpt}"
    out = dict(epoch=EP, ckpt=ckpt, seed=SEED, sigma_dL_frames=SIGMA_DL_FRAMES_PRIMARY,
               run_seeds=RUN_SEEDS, nt=NT, md5_TBW_before=md5_tbw_b, md5_SBW_before=md5_sbw_b)

    # ---------- canonical MSI rate (EVAL path) — jitter-trained ckpt + knob=0 baseline ----------
    P(f"\n=== canonical MSI rate (EVAL path = run_sc_diagnostics bimodal probe; real Hz) ===")
    msi_hz, rates, ie, ep_l = canonical_msi_hz(ckpt, device, f"dL3 ep{EP}", P)
    P(f"  dL3 ep{EP}: MSI={msi_hz:.2f} Hz  I/E={ie:.3f}  all_rates={ {k: round(v,2) for k,v in rates.items()} }")
    base_hz = base_rates = base_ie = None
    if EP == 30:
        base_hz, base_rates, base_ie, _ = canonical_msi_hz(BASELINE_EP30, device, "knob=0 ep30", P)
        P(f"  knob=0 ep30 BASELINE: MSI={base_hz:.2f} Hz  I/E={base_ie:.3f}")
        ratio = (msi_hz / base_hz) if base_hz else float("nan")
        P(f"  --> jitter/baseline MSI rate ratio = {ratio:.3f}  (Lead bar: KILL if >60 Hz AND climbing)")
        out["msi_rate"] = dict(dL3_hz=msi_hz, baseline_hz=base_hz, ratio=ratio,
                               dL3_IE=ie, baseline_IE=base_ie, dL3_rates=rates, baseline_rates=base_rates)
    else:
        out["msi_rate"] = dict(dL3_hz=msi_hz, dL3_IE=ie, dL3_rates=rates)

    # ---------- load the measurement net (fresh; EI before any inference_mode forward, #83) ----------
    net, res, epoch, g_rec, comment = V.load_ckpt(ckpt, SEED, NT, device)
    tau_gaba, tau_nmda_inh, grec = float(net.tau_gaba), float(net.tau_nmda_inh), float(net.g_rec)
    strict_clean = (not res.missing_keys and not res.unexpected_keys)
    sdl_net = float(getattr(net, "sigma_dL_frames", -1.0))
    P(f"\n[dev-check ep{EP}] G0: ep={epoch} tau_gaba={tau_gaba} tau_nmda_inh={tau_nmda_inh} g_rec={grec} "
      f"net.sigma_dL_frames={sdl_net} strict={strict_clean} comment={comment!r}")
    assert strict_clean, "ckpt did not load strict-clean"
    assert abs(tau_gaba - 10.0) < 1e-9 and abs(tau_nmda_inh - 21.6) < 1e-9, "tau mismatch vs reference"
    assert sdl_net == 0.0, "measurement net.sigma_dL_frames must be 0 (jitter enters via the readout arg)"
    assert epoch == EP, f"ckpt epoch {epoch} != requested {EP}"

    # ---- EI FIRST (its in-place reset must precede ANY inference_mode forward, #83) ----
    ei = V.measure_ei(net)
    ei_sync, ei_off = float(ei["ei_ratio_sync"]), float(ei["ei_ratio_offsetmean"])
    rel_off = abs(ei_off - EI_OFF_BASE) / EI_OFF_BASE
    P(f"[dev-check ep{EP}] EI: sync={ei_sync:.4f} offmean={ei_off:.4f} |d off|/{EI_OFF_BASE}={rel_off:.3f}  "
      f"[band sync[0.80,1.25] rel_off<=0.20]")

    # ---- TBW-with-jitter (INTRINSIC), primary sigma_dL=30 ms, pooled M=5 ----
    P(f"\n=== TBW-with-jitter (INTRINSIC sigma_dL={SIGMA_DL_FRAMES_PRIMARY*10:.0f}ms, pooled "
      f"run_seeds={RUN_SEEDS} -> {len(RUN_SEEDS)*NT} trials/SOA) ===")
    units, per_rs, all_eff = [], [], []
    for rs in RUN_SEEDS:
        tk = time.time()
        o, f, n, eff = tbw_fused_counts_jit(net, SIGMA_DL_FRAMES_PRIMARY, run_seed=rs, nt=NT)
        units.append((o, f, n)); all_eff.append(eff)
        mr = G88.graded_metrics(o, f / n)
        per_rs.append(dict(run_seed=rs, P0=float(mr["P0"]), N_inter=int(mr["N_inter"]),
                           max_step=float(mr["max_step"]), FWHM=float(mr["FWHM"])))
        P(f"  [rs={rs}] {time.time()-tk:.1f}s P0={mr['P0']:.3f} N_inter={mr['N_inter']} "
          f"max_step={mr['max_step']:.3f} FWHM={mr['FWHM']:.1f}")
    offs, pf_pool, tf, tn = G88.pool(units)
    m = G88.graded_metrics(offs, pf_pool)
    bm = band_metrics(offs, pf_pool)
    eff_sd_ms = float(np.concatenate(all_eff).std(ddof=1) * 10.0)

    # ---- SBW LAST (frozen readout; sigma_dL does not enter it at inference -> ckpt baseline) ----
    with H.seeded_default_rng(0):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        sbw = V.measure_sbw(net, n_trials=NT)
    sbw_hw, sbw_in = float(sbw["halfwidth_deg"]), bool(sbw["in_band_24p5_40p9"])
    P(f"[dev-check ep{EP}] SBW halfwidth={sbw_hw:.2f} deg in_band[24.5,40.9]={sbw_in} "
      f"Pf={min(sbw['pfusion']):.2f}..{max(sbw['pfusion']):.2f}")

    # ---- fusion@SOA 0 / ±200 ms (read off the jittered pooled curve) ----
    idx = {int(round(o)): i for i, o in enumerate(offs)}
    def pf_at(ms):
        return float(pf_pool[idx[ms]]) if ms in idx else float("nan")
    fus = {"-200": pf_at(-200), "0": pf_at(0), "+200": pf_at(200)}
    P(f"[dev-check ep{EP}] fusion@SOA: -200ms={fus['-200']:.3f}  0ms={fus['0']:.3f}  +200ms={fus['+200']:.3f}")

    # ---- integrity + score + KILL logic ----
    md5_tbw_a, md5_sbw_a = md5s()
    md5_stable = (md5_tbw_a == TBW_MD5 and md5_sbw_a == SBW_MD5)
    finite = bool(np.all(np.isfinite(pf_pool)) and np.isfinite(m["FWHM"]))
    integrity = bool(md5_stable and strict_clean and abs(tau_gaba - 10.0) < 1e-9
                     and abs(tau_nmda_inh - 21.6) < 1e-9 and finite and m["peak"] > 0.0)
    sc = G88.score(m, ei_sync, ei_off, integrity)
    K1 = bool(sc["K"]["K1_box_persists"])
    max_step = float(m["max_step"])
    # Lead ep50 kill: TBW-with-jitter is a hard box -> KILL->debugger if max_step>=0.70 OR K1_box_persists
    ep50_KILL = bool(max_step >= 0.70 or K1)

    curve_str = " ".join(f"{int(round(o)):+d}:{pf_pool[i]:.2f}" for i, o in enumerate(offs) if -160 <= o <= 160)

    out.update(dict(epoch_loaded=epoch, tau_gaba=tau_gaba, tau_nmda_inh=tau_nmda_inh, g_rec=grec,
                    strict_clean=strict_clean, net_sigma_dL_frames=sdl_net,
                    ei_sync=ei_sync, ei_off=ei_off, ei_rel_off=rel_off,
                    sbw_halfwidth_deg=sbw_hw, sbw_in_band=sbw_in,
                    offs_ms=offs.tolist(), pf_pooled=pf_pool.tolist(), fused_pooled=tf.tolist(),
                    trials_pooled=tn.tolist(), per_run_seed=per_rs, eff_offset_sd_ms=eff_sd_ms,
                    metrics={k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                             for k, v in m.items() if k != "pf"}, band=bm,
                    fusion_at_soa=fus, score=sc, integrity=integrity, md5_stable=md5_stable,
                    md5_TBW_after=md5_tbw_a, md5_SBW_after=md5_sbw_a,
                    K1_box_persists=K1, max_step=max_step, ep50_KILL=ep50_KILL,
                    wall_s=time.time() - t0))
    out_json = os.path.join(CKPT_DIR, "out", f"measure_develop_check_ep{EP}.json")
    json.dump(out, open(out_json, "w"), indent=1)

    # ---------------------------------- scorecard ----------------------------------
    P("\n" + "=" * 100)
    P(f"DEVELOP-CHECK POST-HOC — seed42 tau10 ep{EP} (jitter-RETRAIN ckpt) @sigma_dL={SIGMA_DL_FRAMES_PRIMARY*10:.0f}ms "
      f"(INTRINSIC; pooled {len(RUN_SEEDS)}x{NT}/SOA)")
    P("=" * 100)
    P(f"  eff-offset SD (mechanism live) : {eff_sd_ms:.1f} ms  (target {SIGMA_DL_FRAMES_PRIMARY*10:.0f})")
    P(f"  TBW curve [-160..+160]: {curve_str}")
    P(f"      {'metric':<18}{'value':>10}    band / kill")
    P(f"      {'P0':<18}{m['P0']:>10.3f}    >=0.95")
    P(f"      {'N_inter(0.2,0.8)':<18}{m['N_inter']:>10d}    pos>=1 & neg>=1 (pos={m['N_inter_pos']} neg={m['N_inter_neg']})")
    P(f"      {'N_inter(0.1,0.9)':<18}{bm['N_inter_0109']:>10d}    >=4 (graded shorthand)")
    P(f"      {'max_step':<18}{max_step:>10.3f}    <=0.50 GO ; >=0.70 KILL")
    P(f"      {'prom_max':<18}{m['prom_max']:>10.3f}    <=0.10")
    P(f"      {'FWHM(ms)':<18}{m['FWHM']:>10.1f}    [200,300] GO ; WATCH (KILL outside [180,320])")
    P(f"      {'tail':<18}{m['tail']:>10.3f}    <=0.20")
    P(f"      {'range':<18}{m['rng']:>10.3f}    >=0.60")
    P(f"      {'peak':<18}{m['peak']:>10.3f}    >=0.95")
    P(f"  EI sync={ei_sync:.3f} [0.80,1.25]  off={ei_off:.3f} rel={rel_off:.3f} [<=0.20]")
    P(f"  SBW hw={sbw_hw:.2f} deg  in_band[24.5,40.9]={sbw_in}")
    P(f"  fusion@SOA  -200ms={fus['-200']:.3f}  0ms={fus['0']:.3f}  +200ms={fus['+200']:.3f}")
    if EP == 30 and base_hz is not None:
        P(f"  MSI rate (eval) dL3={msi_hz:.2f} Hz  knob=0 baseline={base_hz:.2f} Hz  ratio={msi_hz/base_hz:.3f}")
    else:
        P(f"  MSI rate (eval) dL3={msi_hz:.2f} Hz")
    P("  --- K (any True => NO-GO direction) ---")
    for k, v in sc["K"].items():
        P(f"     [{'TRIP' if v else ' ok '}] {k}")
    P(f"  --- ep50 KILL rule (max_step>=0.70 OR K1_box_persists) ---")
    P(f"     max_step={max_step:.3f} (>=0.70? {max_step>=0.70})  K1_box_persists={K1}  ==>  ep50_KILL={ep50_KILL}")
    P(f"  integrity={integrity}  md5_stable={md5_stable}  finite={finite}")
    md5_tbw_f, md5_sbw_f = md5s()
    P(f"  frozen readouts AFTER: TBW={md5_tbw_f} SBW={md5_sbw_f}  "
      f"(unchanged: {md5_tbw_f==TBW_MD5 and md5_sbw_f==SBW_MD5})")
    assert md5_tbw_f == TBW_MD5 and md5_sbw_f == SBW_MD5, "readout md5 drift AFTER run!"
    P(f"  total {time.time()-t0:.1f}s -> {out_json}")
    P("=" * 100)
    LOG.close()


if __name__ == "__main__":
    main()
