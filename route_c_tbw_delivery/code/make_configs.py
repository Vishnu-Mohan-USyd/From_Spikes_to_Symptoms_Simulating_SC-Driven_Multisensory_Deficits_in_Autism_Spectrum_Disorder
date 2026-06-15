#!/usr/bin/env python
"""make_configs.py — emit the Phase-2 inference-screen override JSONs.

Writes into --out_dir:
  jointA.json            : headline joint-corrected, gNMDA UNCHANGED (0.7)
  jointB.json            : headline joint-corrected, gNMDA driving-force-rescaled
                           (only if --v_rep given; g1 = 0.7*(20-V)/(0-V))
  dose_<param>_<val>.json: one param varied around the joint config (dose-response)

Baseline is run with NO --override_json (true no-op), so no baseline.json needed.
"""
import os, json, argparse

# joint-corrected base (Phase-1 primary-source values); gNMDA set per-variant.
JOINT = dict(tau_nmda_inh=90.0, tau_gaba=20.0, Erev_nmda=0.0, msi_inh2exc_ms=5.0)
BASE_GNMDA = 0.7            # ckpt value (verified across 5 seeds)
EREV_OLD, EREV_NEW = 20.0, 0.0

# dose grids (lead spec). Value == joint is included for a complete table.
DOSES = {
    "msi_inh2exc_ms": [3.0, 4.0, 5.0],     # Whyland & Bickford 2018 (inh->exc IPSC leg)
    "tau_gaba": [10.0, 20.0, 25.0],         # Kirischuk 2005
    "tau_nmda_inh": [60.0, 75.0, 90.0],     # Booker et al. 2021
    "Erev_nmda": [0.0, 5.0],                # Jahr & Stevens 1990 (0..+5 mV)
}


def rescale_gnmda(g0, erev_old, erev_new, v_rep):
    return float(g0) * (float(erev_old) - float(v_rep)) / (float(erev_new) - float(v_rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--v_rep", type=float, default=None,
                    help="measured representative V_msi (mV) from baseline; enables jointB.")
    ap.add_argument("--doses", action="store_true", help="also emit dose-response configs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    written = []

    # jointA — gNMDA unchanged
    a = dict(JOINT); a["gNMDA"] = BASE_GNMDA
    p = os.path.join(args.out_dir, "jointA.json"); json.dump(a, open(p, "w"), indent=1); written.append((p, a))

    # jointB — gNMDA driving-force-rescaled (requires v_rep)
    if args.v_rep is not None:
        g1 = rescale_gnmda(BASE_GNMDA, EREV_OLD, EREV_NEW, args.v_rep)
        b = dict(JOINT); b["gNMDA"] = round(g1, 4)
        p = os.path.join(args.out_dir, "jointB.json"); json.dump(b, open(p, "w"), indent=1); written.append((p, b))
        print("[jointB] v_rep=%.2fmV -> gNMDA rescaled %.3f -> %.4f" % (args.v_rep, BASE_GNMDA, g1))

    # dose-response (one param varied around jointA; gNMDA held at 0.7)
    if args.doses:
        for param, vals in DOSES.items():
            for v in vals:
                cfg = dict(JOINT); cfg["gNMDA"] = BASE_GNMDA; cfg[param] = v
                name = "dose_%s_%s.json" % (param, str(v).replace(".0", "").replace(".", "p"))
                p = os.path.join(args.out_dir, name); json.dump(cfg, open(p, "w"), indent=1); written.append((p, cfg))

    for p, cfg in written:
        print("WROTE %s  %s" % (os.path.basename(p), cfg))
    print("TOTAL %d config(s) in %s" % (len(written), args.out_dir))


if __name__ == "__main__":
    main()
