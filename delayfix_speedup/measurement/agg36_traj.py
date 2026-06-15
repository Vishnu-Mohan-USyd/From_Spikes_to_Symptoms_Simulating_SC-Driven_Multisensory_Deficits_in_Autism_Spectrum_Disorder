#!/usr/bin/env python3
"""TASK #45 trajectory aggregator — build the per-epoch TBW/SBW/E-I table for one seed.

Read-only. Globs out/s{SEED}_ep*.json (one per measured checkpoint), sorts by epoch, and
prints the trajectory the lead asked for: does each observable EVER enter biological range,
and at WHAT epoch does each peel OUT — which observable breaks first (the forensic target).

Usage: agg36_traj.py [SEED=42] [OUT_DIR=out]

Bars: TBW robust-FWHM [396,419] ms (human 414); also report box_width50 (the model-free width
of the is_temporally_fused box, robust when the curve saturates and the Gaussian fit degenerates).
SBW half-width band [24.5,40.9] deg (route-c ref ~31.1). E/I PRIMARY = accumulated MEMBRANE
ratio <I_M>/<I_M_gaba> (ei_ratio_sync/offsetmean), target ~= 1 (debugger #46 ruling); canonical
baseline is mildly inhibition-dominated ~0.46-0.58. comp_ei (~6.34) / panel (13-24) are
source-side, driving-force+tau confounded — NOT membrane balance, excluded from the trajectory.
"""
import json, os, sys, glob

SEED = sys.argv[1] if len(sys.argv) > 1 else "42"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
TBW_FWHM_BAR = (396.0, 419.0)
SBW_BAND = (24.5, 40.9)


def load_all(seed):
    rows = []
    for p in glob.glob(os.path.join(OUT, f"s{seed}_ep*.json")):
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  (skip {os.path.basename(p)}: {e})"); continue
        if d.get("epoch") is None:
            continue
        rows.append(d)
    rows.sort(key=lambda d: int(d["epoch"]))
    return rows


def fmt(x, f="{:.1f}"):
    return f.format(x) if isinstance(x, (int, float)) and x == x else "  -  "


def peeloff(epochs, inrange):
    """First epoch where inrange goes True->False (and stays measured); None if never in / never leaves."""
    ever = any(inrange)
    if not ever:
        return ("NEVER-IN", None)
    # find last contiguous-from-first in-range block end
    last_in = None
    for e, ir in zip(epochs, inrange):
        if ir:
            last_in = e
        else:
            if last_in is not None:
                return ("PEELS-OUT", e)
    return ("STAYS-IN", None)


def main():
    rows = load_all(SEED)
    if not rows:
        print(f"[agg36] no measured checkpoints for seed {SEED} in {OUT} yet.")
        return
    print(f"=== SEED {SEED} TRAJECTORY ({len(rows)} checkpoints) ===")
    print(f"{'ep':>4} {'grec':>4} | {'TBW shape':>12} {'box50(ms)':>10} {'FWHMfit':>8} {'peak':>5} {'P@0':>5} {'FWHMbar?':>8} "
          f"| {'SBW hw°':>7} {'band?':>6} | {'E/I sync':>8} {'E/I off':>8}  (E/I provisional)")
    epochs, tbw_in, sbw_in = [], [], []
    for d in rows:
        ep = int(d["epoch"]); t = d.get("tbw", {}); s = d.get("sbw", {}); e = d.get("ei", {})
        fwhm = t.get("fwhm_ms"); box50 = t.get("box_width50_ms"); peak = t.get("peak"); p0 = t.get("p_at_0")
        fwhm_ok = (fwhm is not None and fwhm == fwhm and TBW_FWHM_BAR[0] <= fwhm <= TBW_FWHM_BAR[1])
        hw = s.get("halfwidth_deg"); band_ok = bool(s.get("in_band_24p5_40p9", False))
        eis = e.get("ei_ratio_sync"); eio = e.get("ei_ratio_offsetmean")
        epochs.append(ep); tbw_in.append(fwhm_ok); sbw_in.append(band_ok)
        print(f"{ep:>4} {fmt(d.get('g_rec'),'{:.1f}'):>4} | {t.get('shape','-'):>12} {fmt(box50):>10} {fmt(fwhm):>8} "
              f"{fmt(peak,'{:.3f}'):>5} {fmt(p0,'{:.2f}'):>5} {('IN' if fwhm_ok else 'out'):>8} "
              f"| {fmt(hw,'{:.2f}'):>7} {('IN' if band_ok else 'out'):>6} "
              f"| {fmt(eis,'{:.3f}'):>8} {fmt(eio,'{:.3f}'):>8}")
    print()
    # peel-off analysis
    print("=== PEEL-OFF (does each observable enter biological range; where does it leave?) ===")
    for name, inr, extra in [("TBW (FWHM in [396,419])", tbw_in, "also see box50 trajectory + shape (box vs graded)"),
                             ("SBW (hw in [24.5,40.9]°)", sbw_in, "")]:
        status, ep = peeloff(epochs, inr)
        line = f"  {name:>26}: {status}" + (f" at ep{ep}" if ep is not None else "")
        if extra:
            line += f"   [{extra}]"
        print(line)
    # E/I = PRIMARY membrane ratio <I_M>/<I_M_gaba> vs ~1 (debugger #46); baseline inh-dom ~0.46-0.58
    eis_traj = [(int(d['epoch']), d.get('ei', {}).get('ei_ratio_sync')) for d in rows]
    eis_traj = [(ep, v) for ep, v in eis_traj if isinstance(v, (int, float)) and v == v]
    if len(eis_traj) >= 2:
        d0, d1 = eis_traj[0][1], eis_traj[-1][1]
        direction = "rising" if d1 > d0 else ("falling" if d1 < d0 else "flat")
        toward = "toward balance(1)" if abs(d1 - 1.0) < abs(d0 - 1.0) else ("away from 1" if abs(d1 - 1.0) > abs(d0 - 1.0) else "unchanged vs 1")
        reaches = any(v >= 0.95 for _, v in eis_traj)
        print(f"  {'E/I membrane (vs ~1)':>26}: {direction}, {toward}  ep{eis_traj[0][0]}={d0:.3f} -> ep{eis_traj[-1][0]}={d1:.3f}  "
              f"(reaches balance >=0.95? {reaches}; baseline inh-dom ~0.46-0.58)")
    elif len(eis_traj) == 1:
        v = eis_traj[0][1]
        print(f"  {'E/I membrane (vs ~1)':>26}: ep{eis_traj[0][0]}={v:.3f}  ({'inh-dominated' if v < 0.95 else 'balanced/exc'} vs target ~1; single point)")
    print()
    # which breaks first
    firsts = []
    for name, inr in [("TBW", tbw_in), ("SBW", sbw_in)]:
        st, ep = peeloff(epochs, inr)
        if st == "PEELS-OUT":
            firsts.append((ep, name))
        elif st == "NEVER-IN":
            firsts.append((-1, name + "(never-in)"))
    if firsts:
        firsts.sort()
        print(f"  -> earliest break: {firsts[0][1]}" + (f" at ep{firsts[0][0]}" if firsts[0][0] >= 0 else " (never entered range)"))
    else:
        print("  -> no peel-off detected among measured epochs (both stay in range or insufficient data)")


if __name__ == "__main__":
    main()
