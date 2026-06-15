#!/usr/bin/env python3
"""TASK #70 A/B comparator — scientific-equivalence of two models through ONE apparatus.

Read-only ANALYSIS (validator). Consumes two val36_traj.py output JSONs:
  --baseline  the BIT-IDENTICAL bs=256 model's curves (= the frozen delayfix science,
              since that model is byte-equal to the frozen delayfix build)
  --test      the RELAXED bs=250 model's curves (4x250 even batch; weights no longer
              byte-equal, so torch.equal is replaced by this curve-equivalence check)

Emits a metric-by-metric comparison (baseline | test | delta | tol | PASS/FAIL) and a
PROPOSED overall verdict against the lead's bar: GO if the bs=250 metrics stay in SIMILAR
RANGE, NO-GO + escalate if ANY curve diverges. Both curves are Monte-Carlo measurements
(n_trials=50), so even IDENTICAL weights differ by ~sqrt(p(1-p)/50) <= 0.071 at p=0.5;
default curve tolerances are set a few MC-sigma wide. ALL thresholds are CLI-overridable
and are PROPOSED pending lead confirmation of the numeric "similar range" bar.

Usage:
  compare_ab.py --baseline out/bs256_ep80.json --test out/bs250_ep80.json [--out_json ...]
                [--tbw_curve_tol 0.15] [--sbw_curve_tol 0.15] [--box50_tol 40]
                [--peak_tol 0.10] [--p0_tol 0.10] [--hw_tol 6.4] [--ei_rel_tol 0.25]
"""
import os, sys, json, argparse
import numpy as np

# ── PRE-REGISTERED #70 tolerances (kill-criteria fixed BEFORE data; all CLI-overridable) ──
# Noise-floor basis: probes are Monte-Carlo (n_trials=50) -> per-point p_fusion SE =
# sqrt(p(1-p)/50) <= 0.071 (worst p=0.5; ~0 on the saturated p in {0,1} plateau/floor).
# Both models are measured at the SAME seed, so stimulus/membrane-noise draws are COMMON-MODE
# and cancel in the delta -> the paired same-seed run-to-run floor is ~1 trial-quantum
# (1/50 = 0.02) at transition bins, ~0 on plateaus. Tolerances are set a few x that floor:
# tight enough to ignore the ~2.3% (256->250) batch wobble, wide enough only for a REAL
# regime change (which moves whole edge bins by >=0.3 / FWHM by >=100 ms / flips E/I across 1).
DEF = dict(
    tbw_curve_tol=0.10,   # max |Δ p_fusion| across the SOA grid (~2.5x the 0.02 paired floor)
    sbw_curve_tol=0.10,   # max |Δ p_fusion| across the separation grid
    box50_tol=40.0,       # |Δ box_width50| ms (<= 2 SOA grid steps of 20 ms = 1-bin edge wobble)
    fwhm_tol=40.0,        # |Δ robust-FWHM| ms
    peak_tol=0.07,        # |Δ peak P(fusion)| (~1 MC-sigma)
    p0_tol=0.07,          # |Δ P(fusion) at SOA 0|
    hw_tol=5.0,           # |Δ SBW half-width| deg (band [24.5,40.9] is 16.4 wide -> ~0.3 band)
    ei_rel_tol=0.20,      # relative |Δ| of membrane E/I ratio (sync & offsetmean)
)


def load(p):
    with open(p) as f:
        return json.load(f)


def _num(x):
    return isinstance(x, (int, float)) and x == x  # finite, non-NaN


def curve_delta(bx, by, tx, ty):
    """Max & mean |Δy| over the x-values present in BOTH curves. Returns (maxd, meand, npts, note)."""
    if not (bx and by and tx and ty):
        return float("nan"), float("nan"), 0, "missing curve"
    bmap = {round(float(x), 6): float(y) for x, y in zip(bx, by) if _num(y)}
    tmap = {round(float(x), 6): float(y) for x, y in zip(tx, ty) if _num(y)}
    shared = sorted(set(bmap) & set(tmap))
    if not shared:
        return float("nan"), float("nan"), 0, "no shared x grid"
    d = np.array([abs(bmap[x] - tmap[x]) for x in shared])
    note = "" if (len(shared) == len(bmap) == len(tmap)) else f"grid overlap {len(shared)}/{max(len(bmap),len(tmap))}"
    return float(d.max()), float(d.mean()), len(shared), note


def row(name, base, test, delta, tol, ok, extra=""):
    def f(v):
        return f"{v:.4f}" if _num(v) else " - "
    flag = "PASS" if ok else ("FAIL" if ok is False else " ?  ")
    print(f"  {name:<34} base={f(base):>9}  test={f(test):>9}  |Δ|={f(delta):>9}  tol={f(tol):>8}  [{flag}] {extra}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="val36 JSON of the bit-identical bs=256 model")
    ap.add_argument("--test", required=True, help="val36 JSON of the relaxed bs=250 model")
    ap.add_argument("--out_json", default=None)
    for k, v in DEF.items():
        ap.add_argument(f"--{k}", type=float, default=v)
    args = ap.parse_args()
    B, Tt = load(args.baseline), load(args.test)
    tol = {k: getattr(args, k) for k in DEF}

    print("=" * 96)
    print("TASK #70 SCIENTIFIC-EQUIVALENCE A/B  (baseline = bit-identical bs=256 | test = relaxed bs=250)")
    print(f"  baseline: {args.baseline}  (epoch={B.get('epoch')} build={B.get('build')} "
          f"tau_nmda_inh={B.get('tau_nmda_inh')} g_rec={B.get('g_rec')})")
    print(f"  test    : {args.test}  (epoch={Tt.get('epoch')} build={Tt.get('build')} "
          f"tau_nmda_inh={Tt.get('tau_nmda_inh')} g_rec={Tt.get('g_rec')})")
    print(f"  PROPOSED tolerances (pending lead sign-off): {tol}")
    print("=" * 96)

    checks, report = [], {}

    # provenance sanity (not gated, but flagged): epoch/build/knobs should match for a fair A/B
    for k in ("epoch", "build", "tau_nmda_inh", "g_rec"):
        if B.get(k) != Tt.get(k):
            print(f"  [WARN] provenance mismatch {k}: baseline={B.get(k)} test={Tt.get(k)} "
                  f"(A/B only fair if these match)")

    # ── TBW ──
    bt, tt = B.get("tbw"), Tt.get("tbw")
    if bt and tt:
        print("\nTBW (temporal binding window):")
        mx, mn, npts, note = curve_delta(bt.get("offsets_ms"), bt.get("pfusion"),
                                         tt.get("offsets_ms"), tt.get("pfusion"))
        report["tbw_curve_maxabs"] = mx; report["tbw_curve_meanabs"] = mn
        checks.append(row("p_fusion curve max|Δ|", 0.0, mx, mx, tol["tbw_curve_tol"],
                          (mx <= tol["tbw_curve_tol"]) if _num(mx) else None, f"({npts} pts {note})"))
        for nm, key, tk in [("box_width50 (ms)", "box_width50_ms", "box50_tol"),
                            ("robust FWHM (ms)", "fwhm_ms", "fwhm_tol"),
                            ("peak P(fusion)", "peak", "peak_tol"),
                            ("P(fusion) @ SOA 0", "p_at_0", "p0_tol")]:
            b, t = bt.get(key), tt.get(key)
            d = abs(b - t) if (_num(b) and _num(t)) else float("nan")
            checks.append(row(nm, b, t, d, tol[tk], (d <= tol[tk]) if _num(d) else None))
        bshape, tshape = bt.get("shape"), tt.get("shape")
        sh_ok = (bshape == tshape)
        print(f"  {'curve shape (qual)':<34} base={bshape!s:>9}  test={tshape!s:>9}  "
              f"{'':>16}  [{'PASS' if sh_ok else 'FAIL'}]")
        checks.append(sh_ok); report["tbw_shape_match"] = sh_ok
        # qualitative regime: central fusion intact (P(fusion)@SOA0 still high)
        tp0 = tt.get("p_at_0"); fus_ok = (tp0 >= 0.5) if _num(tp0) else None
        print(f"  {'fusion present @SOA0 (qual)':<34} {'':>9}  test={tp0 if _num(tp0) else '-'!s:>9}  "
              f"{'thr>=0.5':>27}  [{'PASS' if fus_ok else ('FAIL' if fus_ok is False else ' ? ')}]")
        checks.append(fus_ok); report["tbw_fusion_at0_present"] = fus_ok
    else:
        print("\nTBW: (one side missing — not compared)")

    # ── SBW ──
    bs, ts = B.get("sbw"), Tt.get("sbw")
    if bs and ts:
        print("\nSBW (spatial binding window):")
        mx, mn, npts, note = curve_delta(bs.get("separations_deg"), bs.get("pfusion"),
                                         ts.get("separations_deg"), ts.get("pfusion"))
        report["sbw_curve_maxabs"] = mx; report["sbw_curve_meanabs"] = mn
        checks.append(row("p_fusion curve max|Δ|", 0.0, mx, mx, tol["sbw_curve_tol"],
                          (mx <= tol["sbw_curve_tol"]) if _num(mx) else None, f"({npts} pts {note})"))
        b, t = bs.get("halfwidth_deg"), ts.get("halfwidth_deg")
        d = abs(b - t) if (_num(b) and _num(t)) else float("nan")
        checks.append(row("half-width (deg)", b, t, d, tol["hw_tol"], (d <= tol["hw_tol"]) if _num(d) else None))
        bb, tb = bool(bs.get("in_band_24p5_40p9")), bool(ts.get("in_band_24p5_40p9"))
        band_ok = (bb == tb)
        print(f"  {'in band [24.5,40.9]deg (qual)':<34} base={bb!s:>9}  test={tb!s:>9}  "
              f"{'':>16}  [{'PASS' if band_ok else 'FAIL'}]")
        checks.append(band_ok); report["sbw_band_match"] = band_ok
        # qualitative regime: a real (non-degenerate) band present — finite hw AND curve has relief
        tpf = [v for v in (ts.get("pfusion") or []) if _num(v)]
        rng = (max(tpf) - min(tpf)) if tpf else float("nan")
        present = (_num(ts.get("halfwidth_deg")) and _num(rng) and rng >= 0.2)
        print(f"  {'band present/non-degen (qual)':<34} {'':>9}  test={('rng=%.2f'%rng) if _num(rng) else '-'!s:>9}  "
              f"{'hw finite & rng>=0.2':>27}  [{'PASS' if present else 'FAIL'}]")
        checks.append(bool(present)); report["sbw_band_present"] = bool(present)
    else:
        print("\nSBW: (one side missing — not compared)")

    # ── E/I (PRIMARY = membrane ratio <I_M>/<I_M_gaba>, target ~1; relative tol) ──
    be, te = B.get("ei"), Tt.get("ei")
    if be and te:
        print("\nE/I (PRIMARY membrane ratio <I_M>/<I_M_gaba>, target ~1):")
        for nm, key in [("ei_ratio_sync", "ei_ratio_sync"), ("ei_ratio_offsetmean", "ei_ratio_offsetmean")]:
            b, t = be.get(key), te.get(key)
            rel = abs(b - t) / (abs(b) + 1e-12) if (_num(b) and _num(t)) else float("nan")
            checks.append(row(nm + " (rel Δ)", b, t, rel, tol["ei_rel_tol"],
                              (rel <= tol["ei_rel_tol"]) if _num(rel) else None,
                              f"(abs Δ={abs(b-t):.4f})" if (_num(b) and _num(t)) else ""))
        # qualitative regime: both stay the SAME side of balance (1.0) — not flipped exc<->inh
        bsx, tsx = be.get("ei_ratio_sync"), te.get("ei_ratio_sync")
        side_ok = ((bsx < 1.0) == (tsx < 1.0)) if (_num(bsx) and _num(tsx)) else None
        bside = ("inh-dom" if bsx < 1.0 else "exc-dom") if _num(bsx) else "-"
        tside = ("inh-dom" if tsx < 1.0 else "exc-dom") if _num(tsx) else "-"
        print(f"  {'same regime vs balance=1 (qual)':<34} base={bside:>9}  test={tside:>9}  "
              f"{'':>16}  [{'PASS' if side_ok else ('FAIL' if side_ok is False else ' ? ')}]")
        checks.append(side_ok); report["ei_same_regime"] = side_ok
        report["ei_sync_base"] = bsx; report["ei_sync_test"] = tsx
    else:
        print("\nE/I: (one side missing — not compared)")

    # ── verdict ──
    decided = [c for c in checks if c is not None]
    n_fail = sum(1 for c in decided if c is False)
    n_unknown = sum(1 for c in checks if c is None)
    verdict = "GO" if (n_fail == 0 and n_unknown == 0) else ("NO-GO" if n_fail else "INCOMPLETE")
    print("\n" + "=" * 96)
    print(f"PROPOSED VERDICT: {verdict}   ({len(decided)-n_fail}/{len(decided)} checks PASS, "
          f"{n_fail} FAIL, {n_unknown} undecidable)")
    print("  NOTE: tolerances are PROPOSED; the lead owns the final 'similar range' bar. "
          "Any FAIL => escalate (ship the 19-min bit-identical fallback).")
    print("=" * 96)
    report.update(verdict=verdict, n_pass=len(decided) - n_fail, n_fail=n_fail,
                  n_undecidable=n_unknown, tolerances=tol,
                  baseline=args.baseline, test=args.test)
    if args.out_json:
        json.dump(report, open(args.out_json, "w"), indent=1)
        print(f"  wrote {args.out_json}")
    sys.exit(0 if verdict == "GO" else (2 if verdict == "NO-GO" else 3))


if __name__ == "__main__":
    main()
