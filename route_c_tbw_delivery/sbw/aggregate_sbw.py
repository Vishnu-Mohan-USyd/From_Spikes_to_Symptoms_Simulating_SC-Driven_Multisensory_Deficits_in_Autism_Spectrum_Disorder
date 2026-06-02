#!/usr/bin/env python
"""aggregate_sbw.py — combine the 5 per-ckpt SBW jsons for one prefix into
per-seed half-widths + 5-seed aggregate (mean +/- SD), and fit the grandmean
P(fusion) curve (== make_curves grand_hw). fit_pedestal_curve is lineage-
independent, so either code_dir works for the import."""
import os, sys, json, glob, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--eval_app", required=True)
ap.add_argument("--code_dir", required=True)
ap.add_argument("--glob", required=True, help="e.g. .../sbw_asymrc_m*.json")
ap.add_argument("--prefix", required=True)
ap.add_argument("--out_json", required=True)
args = ap.parse_args()

sys.path.insert(0, args.eval_app)
sys.path.insert(0, args.code_dir)
import torch
_orig = torch.load
def _compat(*a, **k):
    k.setdefault("weights_only", False); return _orig(*a, **k)
torch.load = _compat
import SBW_test as S

files = sorted(glob.glob(args.glob))
assert files, "no files match %s" % args.glob
seeds = []; hws = []; pfs = []; wiring = []; seps = None
for f in files:
    d = json.load(open(f))
    seeds.append(d["tag"]); hws.append(d["halfwidth_deg"]); pfs.append(d["pfusion"]); wiring.append(d["wiring"])
    seps = d["separations_deg"]
pfs = np.array(pfs, float)
grand_pf = np.nanmean(pfs, 0)
try:
    _, _, popt = S.fit_pedestal_curve({"separations_deg": np.array(seps, float), "mean_prob": grand_pf})
    grand_hw = float(abs(popt[2]))
except Exception as e:
    grand_hw = float("nan"); print("[AGG] grand fit fail: %r" % e, flush=True)
hws = np.array(hws, float)
out = dict(prefix=args.prefix, n_seeds=len(files), files=files, seeds=seeds, separations_deg=seps,
           per_seed_halfwidth_deg=hws.tolist(),
           hw_mean=float(np.nanmean(hws)), hw_sd=float(np.nanstd(hws)),
           per_seed_pfusion=pfs.tolist(), grandmean_pfusion=grand_pf.tolist(),
           grandmean_halfwidth_deg=grand_hw,
           degenerate=bool(np.nanmax(grand_pf) - np.nanmin(grand_pf) <= 0.3),
           wiring=wiring,
           asymd_delivered_hw_mean=29.83, asymd_delivered_per_seed=[28.44, 29.40, 30.19, 31.38, 29.75],
           ledger39_hw_deg=27.29, ledger39_band_deg=[24.5, 40.9])
json.dump(out, open(args.out_json, "w"), indent=1)
print("=== AGGREGATE %s (%d seeds) ===" % (args.prefix, len(files)), flush=True)
for s, h in zip(seeds, hws):
    print("  %-12s hw=%.2f deg" % (s, h), flush=True)
print("hw_mean=%.3f  hw_sd=%.3f  grand_hw=%.3f  degenerate=%s" %
      (out["hw_mean"], out["hw_sd"], grand_hw, out["degenerate"]), flush=True)
print("grand Pf range %.3f..%.3f" % (np.nanmin(grand_pf), np.nanmax(grand_pf)), flush=True)
print("WROTE %s" % args.out_json, flush=True)
