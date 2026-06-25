#!/usr/bin/env python3
"""TASK #92 — per-seed-fit-then-aggregate ENSEMBLE measurement for the cross-MODEL-seed TBW bell.

Per #87 (LOCKED, anti-artifact firewall — see PREREG_82_bell_criteria.md "TASK #92"): the FINAL
cross-model-seed bell verdict must FIT EACH model seed individually, THEN aggregate parameters
(mean PSS / width ± across-seed SD) — NOT raw Σfused/Σtrials cross-seed trial-pooling. Raw
cross-seed pooling can MANUFACTURE apparent grading purely from between-seed PSS (center) spread:
an ensemble of N hard BOXES with seed-jittered centers pools into an intermediate staircase
(N_inter>0) though every instance is a box. grade_88_noise.pool() stays correct for WITHIN-instance
measurement-RNG pooling (one ckpt × M run-seeds); THIS module is the layer on top.

Design: imports grade_88_noise (side-effect-free — its main() is __main__-guarded; verified by
read) and reuses tbw_fused_counts / pool / graded_metrics / score and V.TBW.fit_psychometric_curve.
grade_88_noise.py is left BYTE-UNCHANGED so its proven noise=0 bit-exactness gate is preserved.

Modes:
  --mode unittest : synthetic ground-truth (locked in PREREG #92) — PROVES the aggregator separates
                    true within-seed grading from pooling-manufactured grading. No GPU, no ckpts.
  --mode demo107  : real-data plumbing exercise on the THREE measured #107 ep79 curves (loaded from
                    their JSONs — zero new compute). 2 jitter (graded) + 1 no-jitter (box). Shows
                    the aggregator on REAL curves: per-seed-fit flags the box instance (E1) and the
                    2-jitter sub-aggregate reproduces #107's borderline-narrow (E3) at ensemble level.
  --mode ensemble : production driver for the CORRECTED 5-seed jitter retrain when it lands (loads
                    ckpts, measures each within-instance, fits, aggregates). Reuses measure_107's
                    verified jitter primitive. Frozen TBW/SBW md5 asserted before+after.

Validator analysis harness; edits NO production code/tests; frozen readouts untouched.
"""
import os, sys, json, time, argparse, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grade_88_noise as G              # side-effect-free import (main() __main__-guarded)
V = G.V                                 # val36_traj_d52 (frozen readouts + fit_psychometric_curve)
H = G.H                                 # harden_85

FWHM_GO = (200.0, 300.0)               # LOCKED per-curve FWHM GO band (PREREG_82/#88); the #89 prereg floor
FWHM_BIO = (185.0, 270.0)              # #108-established biological AV-TBW FWHM-equiv band (Lead-adjudicated);
                                       # 185=narrow-edge IN-BAND. Reported ALONGSIDE the prereg band; the
                                       # locked #89 GB5 [200,300] is NOT edited (flagged verbatim).
GRADED_FRACTION = 0.8                  # E2: >=ceil(0.8*S) seeds individually graded
NT_SYNTH = 250                         # synthetic trials/SOA (== screen pooled depth M=5x50)
OUT_DIR = os.path.join(HERE, "out")
LOG_DIR = os.path.join(HERE, "logs")

# Official #92 ensemble = the corrected-gate 5-seed jitter retrain (#105). seeds 42/43/44 freshly
# measured (--mode ensemble feeder via measure_107_convergence.py, sigma_dL=3); 45/46 REUSE the #107
# JSONs (identical ckpt files checkpoint/ckpt_ep79_seed{45,46}_..._dL3.pt).
OFFICIAL_SPECS = [
    (os.path.join(OUT_DIR, "measure_107_off92_jit_seed42_ep79.json"), "seed42"),
    (os.path.join(OUT_DIR, "measure_107_off92_jit_seed43_ep79.json"), "seed43"),
    (os.path.join(OUT_DIR, "measure_107_off92_jit_seed44_ep79.json"), "seed44"),
    (os.path.join(OUT_DIR, "measure_107_jit_seed45_ep79.json"), "seed45"),
    (os.path.join(OUT_DIR, "measure_107_jit_seed46_ep79.json"), "seed46"),
]

# #107 measured curves (real-data demo source — read-only; produced by measure_107_convergence.py)
M107 = {
    "nojit_seed42": os.path.join(OUT_DIR, "measure_107_nojit_seed42_ep79.json"),
    "jit_seed45":   os.path.join(OUT_DIR, "measure_107_jit_seed45_ep79.json"),
    "jit_seed46":   os.path.join(OUT_DIR, "measure_107_jit_seed46_ep79.json"),
}


# ---------------------------------------------------------------- per-seed fit + record
def fit_params(offs, pf):
    """Fit the frozen Gaussian psychometric → per-seed PSS (center) + width (FWHM=2.355 sigma)."""
    fit = V.TBW.fit_psychometric_curve(np.asarray(offs, float), np.asarray(pf, float), robust_fit=True)
    return dict(PSS=float(fit["mu"]), FWHM=float(fit["fwhm"]), sigma=float(fit["sigma"]),
                tbw=float(fit["tbw"]), amp=float(fit["amp"]), base=float(fit["base"]),
                r2=float(fit["r_squared"]), peak=float(fit["actual_peak"]))


def per_seed_record(offs, fused, trials, ei_sync, ei_off, integrity, tag):
    """ONE model seed: within-instance curve -> #88 metrics + Gaussian fit + per-curve GB/K verdict.
    is_box ⟺ max_step>=0.70 (== #107/#89). graded ⟺ NOT box AND both limbs (GB2) AND finite slope (GB3)."""
    offs = np.asarray(offs, float); fused = np.asarray(fused, np.int64); trials = np.asarray(trials, np.int64)
    pf = fused / trials
    m = G.graded_metrics(offs, pf)
    fp = fit_params(offs, pf)
    sc = G.score(m, ei_sync, ei_off, integrity)
    is_box = bool(m["max_step"] >= 0.70)
    graded = bool((not is_box) and m["N_inter_pos"] >= 1 and m["N_inter_neg"] >= 1 and m["max_step"] <= 0.50)
    return dict(tag=tag, offs=offs, pf=pf, fused=fused, trials=trials,
                PSS=fp["PSS"], FWHM=fp["FWHM"], sigma=fp["sigma"], tbw=fp["tbw"], r2=fp["r2"],
                N_inter=int(m["N_inter"]), N_inter_pos=int(m["N_inter_pos"]), N_inter_neg=int(m["N_inter_neg"]),
                max_step=float(m["max_step"]), P0=float(m["P0"]), prom_max=float(m["prom_max"]),
                peak=float(m["peak"]), is_box=is_box, graded=graded, per_curve_verdict=sc["verdict"],
                ei_sync=float(ei_sync), ei_off=float(ei_off), integrity=bool(integrity))


def _msd(a):
    a = np.asarray(a, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def aggregate(records, fwhm_band=FWHM_GO, graded_frac=GRADED_FRACTION):
    """PER-SEED-FIT-THEN-AGGREGATE (PREREG #92). Verdict founded on per-seed gradedness; raw
    cross-seed pool computed ONLY as the artifact-guard diagnostic, never contributes to a GO."""
    S = len(records)
    PSS = [r["PSS"] for r in records]; FWHM = [r["FWHM"] for r in records]
    Ninter = np.array([r["N_inter"] for r in records], float)
    pss_m, pss_sd = _msd(PSS); fwhm_m, fwhm_sd = _msd(FWHM)
    p0_m, p0_sd = _msd([r["P0"] for r in records]); ms_m, ms_sd = _msd([r["max_step"] for r in records])
    n_box = int(sum(r["is_box"] for r in records))
    n_graded = int(sum(r["graded"] for r in records))

    # raw cross-seed pool — DIAGNOSTIC ONLY (the thing #87 forbids as a verdict source)
    offs, pf_raw, tf, tn = G.pool([(r["offs"], r["fused"], r["trials"]) for r in records])
    m_raw = G.graded_metrics(offs, pf_raw)
    raw_N = int(m_raw["N_inter"]); raw_ms = float(m_raw["max_step"])
    artifact_flag = bool(raw_N >= 2 and float(np.median(Ninter)) <= 1)

    need_graded = int(math.ceil(graded_frac * S))
    E1_no_box = (n_box == 0)
    E2_each_graded = (n_graded >= need_graded)
    E3_width = bool(fwhm_band[0] <= fwhm_m <= fwhm_band[1])
    E5_integrity = all(r["integrity"] for r in records)
    EK2_artifact = artifact_flag
    verdict = "GO" if (E1_no_box and E2_each_graded and E3_width and E5_integrity and not EK2_artifact) else "NO-GO"

    reasons = []
    if not E1_no_box:      reasons.append(f"EK1 box instance(s) present (n_box={n_box})")
    if EK2_artifact:       reasons.append(f"EK2 pooling-manufactured grading (raw_N={raw_N}, median per-seed N_inter={np.median(Ninter):.0f})")
    if not E2_each_graded: reasons.append(f"E2 too few intrinsically-graded seeds ({n_graded}/{S}, need {need_graded})")
    if not E3_width:       reasons.append(f"E3 FWHM_mean={fwhm_m:.1f} outside {fwhm_band}")
    if not E5_integrity:   reasons.append("E5 integrity failure on >=1 seed")

    return dict(
        S=S, verdict=verdict, reasons=reasons,
        PSS_mean=pss_m, PSS_sd=pss_sd, FWHM_mean=fwhm_m, FWHM_sd=fwhm_sd,
        P0_mean=p0_m, P0_sd=p0_sd, max_step_mean=ms_m, max_step_sd=ms_sd,
        n_box=n_box, n_graded=n_graded, need_graded=need_graded,
        E1_no_box=E1_no_box, E2_each_graded=E2_each_graded, E3_width=E3_width,
        E5_integrity=E5_integrity, artifact_flag=artifact_flag,
        raw_pool=dict(N_inter=raw_N, max_step=raw_ms, pf=[float(x) for x in pf_raw]),
        per_seed=[dict(tag=r["tag"], PSS=r["PSS"], FWHM=r["FWHM"], N_inter=r["N_inter"],
                       N_inter_pos=r["N_inter_pos"], N_inter_neg=r["N_inter_neg"],
                       max_step=r["max_step"], P0=r["P0"], is_box=r["is_box"], graded=r["graded"],
                       per_curve_verdict=r["per_curve_verdict"]) for r in records],
    )


def print_aggregate(agg, P):
    P("  per-seed (FIT EACH, then aggregate):")
    P(f"    {'tag':<16}{'PSS':>8}{'FWHM':>8}{'N_int':>7}{'(+/-)':>7}{'maxstep':>9}{'P0':>7}  box graded")
    for r in agg["per_seed"]:
        P(f"    {r['tag']:<16}{r['PSS']:>8.1f}{r['FWHM']:>8.1f}{r['N_inter']:>7d}"
          f"{('+%d/-%d'%(r['N_inter_pos'],r['N_inter_neg'])):>7}{r['max_step']:>9.3f}{r['P0']:>7.3f}"
          f"  {str(r['is_box']):>5} {str(r['graded']):>5}")
    P(f"  AGGREGATE: PSS={agg['PSS_mean']:.1f}±{agg['PSS_sd']:.1f}ms  FWHM={agg['FWHM_mean']:.1f}±{agg['FWHM_sd']:.1f}ms  "
      f"P0={agg['P0_mean']:.3f}±{agg['P0_sd']:.3f}  max_step={agg['max_step_mean']:.3f}±{agg['max_step_sd']:.3f}")
    P(f"  n_box={agg['n_box']}  n_graded={agg['n_graded']}/{agg['S']} (need {agg['need_graded']})")
    P(f"  RAW cross-seed pool (DIAGNOSTIC ONLY): N_inter={agg['raw_pool']['N_inter']}  max_step={agg['raw_pool']['max_step']:.3f}  "
      f"artifact_flag={agg['artifact_flag']}")
    P(f"  E1_no_box={agg['E1_no_box']} E2_each_graded={agg['E2_each_graded']} E3_width={agg['E3_width']} "
      f"E5_integrity={agg['E5_integrity']} | NOT artifact={not agg['artifact_flag']}")
    P(f"  >>> ENSEMBLE VERDICT = {agg['verdict']}" + (("  reasons: " + "; ".join(agg["reasons"])) if agg["reasons"] else ""))


# ---------------------------------------------------------------- synthetic ground-truth (PREREG #92)
def _counts_from_pf(pf, nt=NT_SYNTH):
    pf = np.clip(np.asarray(pf, float), 0.0, 1.0)
    fused = np.round(pf * nt).astype(np.int64)
    return fused, np.full(len(pf), nt, dtype=np.int64)


def synth_box(offs, center, plateau_lo=-100.0, plateau_hi=60.0, floor=0.0):
    """Hard BOX centered at `center`: P=1 on the plateau, floor outside; one-bin cliff (max_step≈1)."""
    offs = np.asarray(offs, float)
    pf = np.where((offs >= center + plateau_lo) & (offs <= center + plateau_hi), 1.0, floor)
    return _counts_from_pf(pf)


def synth_bell(offs, center, fwhm, amp=1.0, base=0.0):
    """Graded Gaussian BELL: base + amp·exp(-(x-center)²/2σ²), σ=fwhm/2.355."""
    offs = np.asarray(offs, float); sigma = fwhm / 2.355
    pf = base + amp * np.exp(-((offs - center) ** 2) / (2.0 * sigma ** 2))
    return _counts_from_pf(pf)


def run_unittest(P):
    """Synthetic ground-truth locked in PREREG #92 — the correctness proof of the aggregator."""
    offs = np.array([o * 10 for o in G.OFFS], float)
    EI_S, EI_O = 1.0, G.EI_OFF_BASE            # healthy E/I so gradedness is the only variable
    ok_all = True

    def build(curves, tags):
        return [per_seed_record(offs, f, n, EI_S, EI_O, True, t) for (f, n), t in zip(curves, tags)]

    # ---- (A) 5 BOXES, seed-jittered centers → raw-pool fakes grading; per-seed-fit refuses ----
    P("\n[UNITTEST A] 5 hard BOXES, centers {-40,-20,0,+20,+40}ms  (EXPECT NO-GO + artifact_flag)")
    cA = [-40, -20, 0, 20, 40]
    recA = build([synth_box(offs, c) for c in cA], [f"box@{c:+d}" for c in cA])
    aggA = aggregate(recA); print_aggregate(aggA, P)
    okA = (aggA["verdict"] == "NO-GO" and aggA["artifact_flag"] is True and aggA["n_box"] == 5
           and all(r["N_inter"] == 0 for r in aggA["per_seed"]) and aggA["raw_pool"]["N_inter"] >= 2
           and aggA["PSS_sd"] > 20.0)
    P(f"  [A] {'PASS' if okA else 'FAIL'}: NO-GO + artifact + 5 boxes + per-seed N_inter=0 + raw fakes grading + PSS spread")
    ok_all &= okA

    # ---- (B) 5 graded BELLS, FWHM≈240, centers≈0 → ensemble GO; guard does NOT false-trip ----
    P("\n[UNITTEST B] 5 graded BELLS FWHM≈240, centers≈0  (EXPECT GO, artifact_flag False)")
    cB = [-4, -2, 0, 2, 4]
    recB = build([synth_bell(offs, c, 240.0) for c in cB], [f"bell240@{c:+d}" for c in cB])
    aggB = aggregate(recB); print_aggregate(aggB, P)
    okB = (aggB["verdict"] == "GO" and aggB["artifact_flag"] is False and aggB["n_box"] == 0
           and aggB["n_graded"] == 5 and 200.0 <= aggB["FWHM_mean"] <= 300.0 and aggB["PSS_sd"] < 15.0)
    P(f"  [B] {'PASS' if okB else 'FAIL'}: GO + no artifact + 5 graded + FWHM∈[200,300] + tight PSS")
    ok_all &= okB

    # ---- (C) 5 graded BELLS, FWHM≈185 (#107 width) → NO-GO on E3 ONLY (not a box) ----
    P("\n[UNITTEST C] 5 graded BELLS FWHM≈185 (the #107 converged width), centers≈0  (EXPECT NO-GO on E3 only)")
    recC = build([synth_bell(offs, c, 185.0) for c in cB], [f"bell185@{c:+d}" for c in cB])
    aggC = aggregate(recC); print_aggregate(aggC, P)
    okC = (aggC["verdict"] == "NO-GO" and aggC["n_box"] == 0 and aggC["n_graded"] == 5
           and aggC["FWHM_mean"] < 200.0 and aggC["artifact_flag"] is False
           and any("E3" in s for s in aggC["reasons"]) and aggC["E1_no_box"] and aggC["E2_each_graded"])
    P(f"  [C] {'PASS' if okC else 'FAIL'}: NO-GO via E3 only, every seed graded (is_box=False), no artifact")
    ok_all &= okC

    P("\n" + "=" * 96)
    P(f"#92 UNITTEST {'ALL PASS' if ok_all else 'FAILED'} — per-seed-fit-then-aggregate separates true "
      f"within-seed grading from pooling-manufactured grading.")
    P("=" * 96)
    json.dump(dict(mode="unittest", okA=bool(okA), okB=bool(okB), okC=bool(okC), ok_all=bool(ok_all),
                   A=aggA, B=aggB, C=aggC), open(os.path.join(OUT_DIR, "ensemble_92_unittest.json"), "w"),
              indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    return ok_all


# ---------------------------------------------------------------- real-data demo on the #107 curves
def _record_from_107(path, tag):
    d = json.load(open(path))
    offs = np.asarray(d["offs_ms"], float)
    fused = np.asarray(d["fused_pooled"], np.int64); trials = np.asarray(d["trials_pooled"], np.int64)
    integ = bool(d.get("md5_stable", False) and d.get("strict_clean", True)
                 and d.get("md5_TBW_after") == G.TBW_MD5 and float(d.get("net_sigma_dL_frames", 0.0)) == 0.0)
    return per_seed_record(offs, fused, trials, float(d["ei_sync"]), float(d["ei_off"]), integ, tag)


def run_demo107(P):
    """Exercise the aggregator on the THREE REAL measured #107 curves (no new compute)."""
    missing = [k for k, p in M107.items() if not os.path.exists(p)]
    if missing:
        P(f"[demo107] SKIP — missing #107 artifacts: {missing}"); return None
    recs = [_record_from_107(M107["jit_seed45"], "jit_seed45"),
            _record_from_107(M107["jit_seed46"], "jit_seed46"),
            _record_from_107(M107["nojit_seed42"], "nojit_seed42")]

    P("\n" + "=" * 96)
    P("#92 DEMO on REAL #107 curves — MIXED set (2 jitter + 1 no-jitter); plumbing/real-data exercise,")
    P("NOT a scientific 5-seed ensemble. Expect: the no-jitter box instance trips E1 -> NO-GO.")
    P("=" * 96)
    aggMix = aggregate(recs); print_aggregate(aggMix, P)

    P("\n  --- 2-JITTER sub-aggregate (seed45+seed46 only): real-data analog of UNITTEST C ---")
    aggJit = aggregate(recs[:2]); print_aggregate(aggJit, P)

    json.dump(dict(mode="demo107", mixed=aggMix, jitter_only=aggJit),
              open(os.path.join(OUT_DIR, "ensemble_92_demo107.json"), "w"),
              indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    return dict(mixed=aggMix, jitter_only=aggJit)


# ---------------------------------------------------------------- OFFICIAL 5-seed ensemble verdict
def _official_record(path, tag):
    """Load one per-seed measure_107 JSON (TBW pooled curve + SBW + EI + MSI) and build the record.
    integrity REQUIRES the frozen TBW md5 + sigma_dL_frames=0 + strict load that JSON recorded."""
    d = json.load(open(path))
    offs = np.asarray(d["offs_ms"], float)
    fused = np.asarray(d["fused_pooled"], np.int64); trials = np.asarray(d["trials_pooled"], np.int64)
    integ = bool(d.get("md5_stable", False) and d.get("strict_clean", True)
                 and d.get("md5_TBW_after") == G.TBW_MD5 and float(d.get("net_sigma_dL_frames", 0.0)) == 0.0)
    r = per_seed_record(offs, fused, trials, float(d["ei_sync"]), float(d["ei_off"]), integ, tag)
    r["sbw_hw"] = float(d.get("sbw_halfwidth_deg", float("nan")))
    r["sbw_in_band"] = bool(d.get("sbw_in_band", False))
    r["msi_hz"] = float(d.get("msi_hz", float("nan")))
    r["ei_rel_off"] = abs(r["ei_off"] - G.EI_OFF_BASE) / G.EI_OFF_BASE
    r["ei_in_band"] = bool(0.80 <= r["ei_sync"] <= 1.25 and r["ei_rel_off"] <= 0.20)   # #89 GB7
    r["ckpt"] = d.get("ckpt", path)
    return r


def run_official(specs, P):
    """OFFICIAL per-seed-fit-then-aggregate pooled GO/NO-GO on the corrected-gate 5-seed ensemble.
    FWHM judged TWO ways (both reported): (a) #89 prereg [200,300] verbatim; (b) #108 biological
    [185,270]. SBW + E/I + MSI reported as vitals. Frozen-readout integrity inherited per-seed."""
    missing = [p for p, _ in specs if not os.path.exists(p)]
    if missing:
        P(f"[official] BLOCKED — missing per-seed artifacts:\n   " + "\n   ".join(missing)); return None
    recs = [_official_record(p, t) for p, t in specs]
    aggP = aggregate(recs, fwhm_band=FWHM_GO)     # as-preregistered #89 floor
    aggB = aggregate(recs, fwhm_band=FWHM_BIO)    # #108 biological band

    sbw_m, sbw_sd = _msd([r["sbw_hw"] for r in recs]); sbw_inb = int(sum(r["sbw_in_band"] for r in recs))
    eis_m, eis_sd = _msd([r["ei_sync"] for r in recs]); eio_m, eio_sd = _msd([r["ei_off"] for r in recs])
    ei_inb = int(sum(r["ei_in_band"] for r in recs))
    msi_m, msi_sd = _msd([r["msi_hz"] for r in recs]); msi_flag = int(sum(1 for r in recs if r["msi_hz"] < 5.0))
    per_curve_go = int(sum(1 for r in recs if r["per_curve_verdict"] == "GO"))
    S = len(recs)

    P("\n" + "=" * 100)
    P(f"#92 OFFICIAL ENSEMBLE — corrected-gate 5-seed jitter retrain (ep79, sigma_dL=30ms), per-seed-fit-then-aggregate")
    P("=" * 100)
    print_aggregate(aggP, P)
    P("  ---- FWHM judged TWO ways (locked #89 rubric NOT edited; both reported verbatim) ----")
    P(f"    (a) #89 AS-PREREGISTERED  GB5 FWHM∈[200,300]:  FWHM_mean={aggP['FWHM_mean']:.1f} -> E3={aggP['E3_width']}  => ensemble {aggP['verdict']}")
    P(f"    (b) #108 BIOLOGICAL band  FWHM∈[185,270]:      FWHM_mean={aggB['FWHM_mean']:.1f} -> E3={aggB['E3_width']}  => ensemble {aggB['verdict']}")
    P(f"        (units guard: Gaussian-FWHM vs FWHM-equiv; NOT vs 50%-criterion ~280-461 nor SC ~250-700)")
    P("  ---- vitals (aggregate ± across-seed SD) ----")
    P(f"    SBW halfwidth = {sbw_m:.2f}±{sbw_sd:.2f}°  in-band[24.5,40.9]: {sbw_inb}/{S}")
    P(f"    E/I sync = {eis_m:.3f}±{eis_sd:.3f}  off = {eio_m:.3f}±{eio_sd:.3f}  GB7 in-band: {ei_inb}/{S}")
    P(f"    canonical MSI = {msi_m:.2f}±{msi_sd:.2f} Hz  (<5Hz flagged: {msi_flag}/{S})")
    P(f"    per-seed full #89 per-curve verdict GO: {per_curve_go}/{S}")
    P("  ---- HEADLINE ----")
    P(f"    Box->bell: n_box={aggP['n_box']}/{S}  n_graded={aggP['n_graded']}/{S}  artifact_flag={aggP['artifact_flag']}")
    P(f"    ENSEMBLE VERDICT: {aggB['verdict']} under #108 biological FWHM band; {aggP['verdict']} under #89 as-prereg [200,300] floor.")
    P("=" * 100)

    out = dict(mode="official", specs=[dict(path=p, tag=t) for p, t in specs],
               prereg_FWHM_band=FWHM_GO, biological_FWHM_band=FWHM_BIO,
               agg_prereg=aggP, agg_biological=aggB,
               sbw=dict(mean=sbw_m, sd=sbw_sd, in_band=sbw_inb, n=S),
               ei=dict(sync_mean=eis_m, sync_sd=eis_sd, off_mean=eio_m, off_sd=eio_sd, in_band=ei_inb, n=S),
               msi=dict(mean=msi_m, sd=msi_sd, flagged_lt5=msi_flag, n=S),
               per_curve_go=per_curve_go,
               per_seed_vitals=[dict(tag=r["tag"], sbw_hw=r["sbw_hw"], sbw_in_band=r["sbw_in_band"],
                                     ei_sync=r["ei_sync"], ei_off=r["ei_off"], ei_in_band=r["ei_in_band"],
                                     msi_hz=r["msi_hz"], per_curve_verdict=r["per_curve_verdict"]) for r in recs])
    json.dump(out, open(os.path.join(OUT_DIR, "ensemble_92_official.json"), "w"),
              indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    return out


# ---------------------------------------------------------------- production ensemble driver (future)
def run_ensemble(ckpt_specs, sigma_dL, run_seeds, nt, P, device="cuda"):
    """CORRECTED 5-seed jitter retrain path. ckpt_specs = [(ckpt_path, seed, tag), ...].
    Measures each seed WITHIN-instance (jitter primitive from measure_107), fits, aggregates.
    Frozen TBW/SBW md5 asserted before + after. RUN only when corrected ckpts exist."""
    import measure_107_convergence as M107C            # reuse the verified jitter measurement primitive
    md5_tbw_b, md5_sbw_b = G.md5s()
    assert md5_tbw_b == G.TBW_MD5 and md5_sbw_b == G.SBW_MD5, "frozen readout md5 drift BEFORE ensemble!"
    recs = []
    for ckpt, seed, tag in ckpt_specs:
        net, res, epoch, g_rec, comment = V.load_ckpt(ckpt, seed, nt, device)
        strict = (not res.missing_keys and not res.unexpected_keys)
        assert abs(float(net.tau_nmda_inh) - 21.6) < 1e-9 and float(getattr(net, "sigma_dL_frames", 0.0)) == 0.0
        ei = V.measure_ei(net)
        units = [M107C.MDC.tbw_fused_counts_jit(net, sigma_dL, rs, nt) for rs in run_seeds]
        offs, _pf, tf, tn = G.pool(units)             # within-instance pool over measurement run-seeds
        integ = bool(strict and abs(float(net.tau_gaba) - 10.0) < 1e-9)
        recs.append(per_seed_record(offs, tf, tn, float(ei["ei_ratio_sync"]),
                                    float(ei["ei_ratio_offsetmean"]), integ, tag))
        P(f"  measured {tag}: ep{epoch} N_inter={recs[-1]['N_inter']} max_step={recs[-1]['max_step']:.3f} "
          f"FWHM={recs[-1]['FWHM']:.1f} is_box={recs[-1]['is_box']}")
    md5_tbw_a, md5_sbw_a = G.md5s()
    assert md5_tbw_a == G.TBW_MD5 and md5_sbw_a == G.SBW_MD5, "frozen readout md5 drift AFTER ensemble!"
    agg = aggregate(recs); agg["md5_stable"] = True
    print_aggregate(agg, P)
    json.dump(dict(mode="ensemble", sigma_dL=sigma_dL, run_seeds=run_seeds, nt=nt, agg=agg),
              open(os.path.join(OUT_DIR, "ensemble_92_ensemble.json"), "w"),
              indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["unittest", "demo107", "official", "ensemble"], default="unittest")
    ap.add_argument("--sigma_dL", type=float, default=3.0)
    ap.add_argument("--run_seeds", default="0,1,2,3,4")
    ap.add_argument("--nt", type=int, default=G.NT_DEFAULT)
    ap.add_argument("--ckpts", default="", help="ensemble mode: ckpt:seed:tag,ckpt:seed:tag,...")
    ap.add_argument("--log", default=os.path.join(LOG_DIR, "ensemble_92.log"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    LOG = open(args.log, "w", buffering=1)
    def P(*a): print(*a, flush=True); print(*a, file=LOG, flush=True)
    t0 = time.time()
    P(f"[#92 ensemble] mode={args.mode}  TBW.md5={G.md5s()[0]}  FWHM_GO={FWHM_GO}  graded_frac={GRADED_FRACTION}")

    rc = 0
    if args.mode == "unittest":
        rc = 0 if run_unittest(P) else 1
    elif args.mode == "demo107":
        run_demo107(P)
    elif args.mode == "official":
        run_official(OFFICIAL_SPECS, P)
    else:
        run_seeds = [int(x) for x in args.run_seeds.split(",") if x.strip()]
        specs = []
        for tok in args.ckpts.split(","):
            if not tok.strip(): continue
            c, s, t = tok.split(":"); specs.append((c, int(s), t))
        assert specs, "ensemble mode needs --ckpts ckpt:seed:tag,..."
        run_ensemble(specs, args.sigma_dL, run_seeds, args.nt, P)
    P(f"[#92 ensemble] total {time.time()-t0:.1f}s")
    LOG.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
