#!/usr/bin/env python3
"""VALIDATOR #394 GRADER — grade the dm10 ensemble full-7 outputs vs the OFFICIAL paper criteria
(README + per-gate scripts), per-seed AND ensemble mean+/-SD. Pure JSON->verdict; no GPU, no model code.

Official criteria (NOT the aM014 #372 bands):
  G1 RATE      : per-seed MSI in [12,32] Hz
  G2 TBW       : per-seed raw half-max FWHM <= 300 ms (HARD line) and finite
  G3 E/I       : per-seed shunt-aware sync in [0.80,1.25] (biological balance ~1.0)
  G4 POP-IE    : per-seed NEGATIVE MEI-vs-log(I) slope AND MEI_lo(0.05) > MEI_hi(1.6)  [inverse effectiveness]
  G5 SBW       : per-seed 3-sign centre-surround: peak_cre>0 AND zero_cross>0 AND surr_min<0
  G6 LATENCY   : §2.7 mean Delta = mean(L_A,L_V)-L_B > 0 at INTENSITY=1.0 (onset facilitation). The race
                 descriptor min(L_A,L_V)-L_B tying ~0 at strong I is the honest null, NOT a fail. Low-I
                 inverse-effectiveness race benefit is reported as honest CONTEXT, not the gate.
  G7 CUE-REL   : R2_pooled > 0.71 AND every per-seed R2 > 0.71 (gain_exp=1 headline)

Usage: python val394_dm10_grade.py [--dir /tmp/val394_dm10] [--gain_exp 1]
"""
import os, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/tmp/val394_dm10")
ap.add_argument("--gain_exp", type=int, default=1)
A = ap.parse_args()
D = A.dir


def load(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else None


def stat(vals):
    a = np.asarray([v for v in vals if v is not None and v == v], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), (float(np.std(a, ddof=1)) if a.size > 1 else 0.0)


main = load("gates_main.json")
cre = load("gate5_cre.json")
lat = load("gate6_latency_sweep.json")
g7 = load(f"gate7_cuerel_g{A.gain_exp}_aggregate.json")

per = {}        # seed -> {gate: (pass_bool, detail)}
fails = []      # (seed, gate)


def mark(seed, gate, ok, detail):
    per.setdefault(seed, {})[gate] = (bool(ok), detail)
    if not ok:
        fails.append((seed, gate))


# ---------- G1/G2/G3/G4 from gates_main ----------
if main:
    INT = main.get("ie_intensities") or main["rows"][0].get("ie_intensities")
    logI = np.log10(np.asarray(INT, float))
    for r in main["rows"]:
        s = r["seed"]
        rate = r["msi_hz"]; mark(s, "G1_RATE", 12.0 <= rate <= 32.0, f"{rate:.2f}Hz")
        tbw = r["tbw_raw_fwhm"]; mark(s, "G2_TBW", (tbw == tbw) and tbw <= 300.0, f"{tbw:.0f}ms")
        ei = r["ei_sync"]; mark(s, "G3_EI", 0.80 <= ei <= 1.25, f"{ei:.3f}")
        mei = np.asarray(r["mei"], float)
        slope = float(np.polyfit(logI, mei, 1)[0])
        dec = mei[0] > mei[-1]
        amax = INT[int(np.argmax(mei))]
        mark(s, "G4_IE", (slope < 0) and dec,
             f"slope={slope:+.2f} MEIlo={mei[0]:.2f}>MEIhi={mei[-1]:.2f}={dec} peak@{amax:g}")

# ---------- G5 from cre ----------
if cre:
    for r in cre["rows"]:
        s = r["seed"]
        pk = r["peak_cre"]; zc = r["zero_cross"]; sm = r["surr_min"]
        ok = (pk is not None and pk > 0) and (zc is not None and zc > 0) and (sm is not None and sm < 0)
        mark(s, "G5_SBW", ok, f"peak={pk:.1f}% zc={('NaN' if zc is None else round(zc,1))} trough={sm:.1f}%")

# ---------- G6 from latency sweep: §2.7 mean Delta = mean(L_A,L_V)-L_B > 0 at INTENSITY=1.0 ----------
if lat:
    INTL = [float(x) for x in lat["intensities"]]
    j1 = INTL.index(1.0) if 1.0 in INTL else int(np.argmin([abs(x - 1.0) for x in INTL]))
    jlo = int(np.argmin(INTL))
    for r in lat["rows"]:
        s = r["seed"]
        la = float(r["L_A"][j1]); lv = float(r["L_V"][j1]); lb = float(r["L_B"][j1])
        mean_uni = 0.5 * (la + lv)
        delta = mean_uni - lb                          # §2.7 onset-facilitation mean Delta (the GATE)
        race = min(la, lv) - lb                         # race descriptor (tie ~0 at strong I = honest null)
        loben = float(r["benefit"][jlo])                # low-I inverse-effectiveness race benefit (context)
        ok = (delta == delta) and delta > 0
        mark(s, "G6_LAT", ok,
             f"meanD@I=1.0={delta:+.2f}ms [mean(LA,LV)={mean_uni:.1f}-LB={lb:.1f}]; "
             f"race@1.0={race:+.2f}(null-ok); loben@{INTL[jlo]:g}={loben:+.1f}")

# ---------- G7: per-seed R2 from per-seed JSONs (aggregate carries only pooled + mean/sem) ----------
import glob as _glob
g7_line = None
ps_items = []
for pf in sorted(_glob.glob(os.path.join(D, f"gate7_cuerel_g{A.gain_exp}_seed*.json"))):
    rec = json.load(open(pf))
    sk = int(rec["seed"]); v = float(rec["r2"])
    ps_items.append((sk, v))
    mark(sk, "G7_CUE", v > 0.71, f"R2={v:.3f}")
if g7 or ps_items:
    r2p = g7.get("r2_pooled") if g7 else None
    r2min = min((v for _, v in ps_items), default=float("nan"))
    g7_pool_ok = (r2p is not None) and r2p > 0.71
    g7_line = (r2p, r2min, g7_pool_ok, len(ps_items))

# ---------- report ----------
GATES = ["G1_RATE", "G2_TBW", "G3_EI", "G4_IE", "G5_SBW", "G6_LAT", "G7_CUE"]
seeds = sorted(per.keys())
print("\n================ VALIDATOR #394 dm10 GRADE — official criteria ================\n")
print("  seed | " + " | ".join(f"{g:^8}" for g in GATES))
print("  " + "-" * (7 + 11 * len(GATES)))
for s in seeds:
    row = []
    for g in GATES:
        pv = per[s].get(g)
        row.append("  --  " if pv is None else ("  PASS  " if pv[0] else " *FAIL* "))
    print(f"  {s:>4} | " + " | ".join(f"{c:^8}" for c in row))

print("\n  PER-SEED DETAIL:")
for s in seeds:
    for g in GATES:
        pv = per[s].get(g)
        if pv is not None:
            print(f"    s{s} {g:8}: {'PASS' if pv[0] else 'FAIL'}  {pv[1]}")

# ---------- ensemble mean+/-SD ----------
print("\n  ENSEMBLE (mean +/- SD across seeds):")
if main:
    rm, rs = stat([r["msi_hz"] for r in main["rows"]])
    tm, ts = stat([r["tbw_raw_fwhm"] for r in main["rows"]])
    em, es = stat([r["ei_sync"] for r in main["rows"]])
    print(f"    G1 RATE = {rm:.2f}+/-{rs:.2f} Hz   band[12,32]")
    print(f"    G2 TBW  = {tm:.0f}+/-{ts:.0f} ms    HARD<=300")
    print(f"    G3 E/I  = {em:.3f}+/-{es:.3f}     [0.80,1.25]")
    mei_mat = np.array([r["mei"] for r in main["rows"]], float)
    mm = mei_mat.mean(0)
    print(f"    G4 MEI  = [{' '.join(f'{x:.2f}' for x in mm)}] @I={main.get('ie_intensities')}")
if cre:
    pm, psd = stat([r["peak_cre"] for r in cre["rows"]])
    zm, zsd = stat([r["zero_cross"] for r in cre["rows"]])
    sm2, ssd = stat([r["surr_min"] for r in cre["rows"]])
    print(f"    G5 SBW  = peak {pm:.1f}+/-{psd:.1f}%  zero-cross {zm:.1f}+/-{zsd:.1f}deg  trough {sm2:.1f}+/-{ssd:.1f}%")
if lat:
    INTL = [float(x) for x in lat["intensities"]]
    j1 = INTL.index(1.0) if 1.0 in INTL else int(np.argmin([abs(x - 1.0) for x in INTL]))
    jlo = int(np.argmin(INTL))
    deltas = [0.5 * (float(r["L_A"][j1]) + float(r["L_V"][j1])) - float(r["L_B"][j1]) for r in lat["rows"]]
    dm_, ds_ = stat(deltas)
    print(f"    G6 LAT  = mean Delta@I=1.0 {dm_:+.2f}+/-{ds_:.2f} ms   (>0 gate; §2.7 onset facilitation)")
    bpi = lat.get("benefit_per_intensity")
    if bpi:
        print(f"             context: low-I(I={INTL[jlo]:g}) race benefit "
              f"{bpi[jlo]['mean']:+.2f}+/-{bpi[jlo]['sd']:.2f} ms;  race curve {[round(b['mean'],1) for b in bpi]}")
if g7_line:
    r2p, r2min, ok7, n7 = g7_line
    print(f"    G7 CUE  = R2 pooled {r2p:.3f}  per-seed min {r2min:.3f}  (n={n7})  bar>0.71")

# ---------- verdict ----------
present_gates = set(g for s in per for g in per[s])
missing = [g for g in GATES if g not in present_gates]
pooled_fail = []
if g7_line and not g7_line[2]:
    pooled_fail.append("G7_pooled")
verdict = "GO" if (not fails and not pooled_fail and not missing) else "NO-GO"
print("\n  ================ VERDICT ================")
print(f"  {verdict}")
if missing:
    print(f"    INCOMPLETE — no data for: {missing}")
if fails:
    print(f"    per-seed failures: {sorted(fails)}")
if pooled_fail:
    print(f"    pooled failures: {pooled_fail}")
print("  ========================================\n")
