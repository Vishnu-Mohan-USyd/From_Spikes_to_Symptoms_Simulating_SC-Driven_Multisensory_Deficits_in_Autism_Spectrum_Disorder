#!/usr/bin/env python
"""ei_aggregate.py — pool the 5 per-ckpt E/I jsons for ONE condition into a
5-seed mean +/- SD of the route-c E/I ratio (<I_M>/<I_M_gaba>), plus the
component-level cross-check and the representative operating voltage v_rep
(used to drive-force-rescale gNMDA). Paper target E/I ~ 1.04."""
import json, glob, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True, help="e.g. .../ei_asymrc_baseline_m*.json")
ap.add_argument("--prefix", required=True)
ap.add_argument("--out_json", required=True)
args = ap.parse_args()

files = sorted(glob.glob(args.glob))
assert files, "no files match %s" % args.glob
sync, offm, comp, vsync, voff, vwsync, tags = [], [], [], [], [], [], []
for f in files:
    d = json.load(open(f))
    tags.append(d["tag"])
    sync.append(d["ei_ratio_sync"]); offm.append(d["ei_ratio_offsetmean"])
    comp.append(d["comp_ei_ratio_offsetmean"])
    vsync.append(d["v_rep_sync"]); voff.append(d["v_rep_offsetmean"])
    vwsync.append(d.get("v_rep_weighted_sync", float("nan")))  # I_M-weighted (gNMDA rescale)
sync = np.asarray(sync, float); offm = np.asarray(offm, float); comp = np.asarray(comp, float)
vsync = np.asarray(vsync, float); voff = np.asarray(voff, float); vwsync = np.asarray(vwsync, float)

out = dict(
    prefix=args.prefix, n_seeds=len(files), files=files, seeds=tags,
    definition="route-c E/I = <I_M>/<I_M_gaba> time-avg over response window; target ~1.04",
    per_seed_ei_sync=sync.tolist(), per_seed_ei_offsetmean=offm.tolist(),
    ei_sync_mean=float(np.nanmean(sync)), ei_sync_sd=float(np.nanstd(sync)),
    ei_offsetmean_mean=float(np.nanmean(offm)), ei_offsetmean_sd=float(np.nanstd(offm)),
    comp_ei_mean=float(np.nanmean(comp)), comp_ei_sd=float(np.nanstd(comp)),
    v_rep_sync_mean=float(np.nanmean(vsync)), v_rep_offsetmean_mean=float(np.nanmean(voff)),
    v_rep_weighted_sync_mean=float(np.nanmean(vwsync)),  # I_M-weighted -> gNMDA rescale basis for jointB
    per_seed_v_rep_weighted_sync=vwsync.tolist(),
    paper_target_ei=1.04,
)
json.dump(out, open(args.out_json, "w"), indent=1)
print("=== EI AGGREGATE %s (%d seeds) ===" % (args.prefix, len(files)), flush=True)
for t, s in zip(tags, sync):
    print("  %-14s E/I(sync)=%.3f" % (t, s), flush=True)
print("  E/I(sync) mean=%.3f sd=%.3f | E/I(offmean) mean=%.3f sd=%.3f | comp-E/I mean=%.3f" %
      (out["ei_sync_mean"], out["ei_sync_sd"], out["ei_offsetmean_mean"], out["ei_offsetmean_sd"], out["comp_ei_mean"]), flush=True)
print("  v_rep(sync) mean=%.2f mV  v_rep_weighted(sync) mean=%.2f mV  (weighted = gNMDA rescale basis)  | target E/I=1.04" %
      (out["v_rep_sync_mean"], out["v_rep_weighted_sync_mean"]), flush=True)
print("WROTE %s" % args.out_json, flush=True)
