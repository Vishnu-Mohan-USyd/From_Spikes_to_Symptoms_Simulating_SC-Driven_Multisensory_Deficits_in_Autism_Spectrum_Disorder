#!/usr/bin/env python
"""tbw_aggregate.py — pool the 5 per-model TBW npzs for ONE condition into a
grandmean P(fusion) curve + robust Gaussian fit + box/graded discriminator +
per-SOA grand mean #peaks (the Phase-2 peak-count dump, pooled).

Each per-model npz is produced by tbw_routec_curve.py --only_model k and holds
mean_fusion (==that model's P(fusion)) and mean_npeaks. We stack across the 5
seeds. fit_psychometric_curve is lineage-independent."""
import sys, json, glob, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--code_dir", required=True)
ap.add_argument("--glob", required=True, help="e.g. .../asymrc_tbw_ep80_baseline_m*.npz")
ap.add_argument("--prefix", required=True, help="condition label, e.g. baseline / jointA / jointB")
ap.add_argument("--out_json", required=True)
args = ap.parse_args()

sys.path.insert(0, args.code_dir)
from TBW_test import fit_psychometric_curve

files = sorted(glob.glob(args.glob))
assert files, "no files match %s" % args.glob
oms = None
pf, npk = [], []
for f in files:
    d = np.load(f, allow_pickle=True)
    oms = np.asarray(d["offsets_ms"], float)
    pf.append(np.asarray(d["mean_fusion"], float))
    npk.append(np.asarray(d["mean_npeaks"], float))
pf = np.vstack(pf)          # (n_seeds, n_offsets)
npk = np.vstack(npk)        # (n_seeds, n_offsets)
grand_pf = pf.mean(0)
grand_npk = npk.mean(0)


def at(arr, target):
    j = int(np.argmin(np.abs(oms - target)))
    return float(arr[j])


fit = fit_psychometric_curve(oms, grand_pf)
# box vs graded discriminator: count strictly-intermediate P (0.02<P<0.98)
n_graded = int(np.sum((grand_pf > 0.02) & (grand_pf < 0.98)))
is_box = bool(np.nanmin(grand_pf) > 0.9)   # flat-top ~1 everywhere

out = dict(
    prefix=args.prefix, n_seeds=len(files), files=files,
    offsets_ms=oms.tolist(),
    grandmean_pfusion=grand_pf.tolist(),
    per_seed_pfusion=pf.tolist(),
    grandmean_npeaks=grand_npk.tolist(),
    per_seed_npeaks=npk.tolist(),
    Pfus_at_0=at(grand_pf, 0), Pfus_at_m200=at(grand_pf, -200), Pfus_at_p200=at(grand_pf, 200),
    peak_Pfus=float(grand_pf.max()), min_Pfus=float(grand_pf.min()),
    npeaks_at_0=at(grand_npk, 0), npeaks_at_m200=at(grand_npk, -200), npeaks_at_p200=at(grand_npk, 200),
    npeaks_min=float(grand_npk.min()), npeaks_max=float(grand_npk.max()),
    fwhm_ms=float(fit["fwhm"]), sigma_ms=float(fit["sigma"]), mu_ms=float(fit["mu"]),
    tbw_ms=float(fit["tbw"]), r_squared=float(fit["r_squared"]),
    n_graded=n_graded, n_offsets=int(len(oms)),
    is_box_flat=is_box,
    human_targets=dict(fwhm_ms=414, sdrob_ms=176, tbw_paper_ms=215),
)
json.dump(out, open(args.out_json, "w"), indent=1)
print("=== TBW AGGREGATE %s (%d seeds) ===" % (args.prefix, len(files)), flush=True)
print("  P(fus) @0=%.3f  -200=%.3f  +200=%.3f  peak=%.3f  min=%.3f" %
      (out["Pfus_at_0"], out["Pfus_at_m200"], out["Pfus_at_p200"], out["peak_Pfus"], out["min_Pfus"]), flush=True)
print("  npeaks @0=%.2f  -200=%.2f  +200=%.2f  (min=%.2f max=%.2f)" %
      (out["npeaks_at_0"], out["npeaks_at_m200"], out["npeaks_at_p200"], out["npeaks_min"], out["npeaks_max"]), flush=True)
print("  FWHM=%.1fms SDrob=%.1fms mu=%.1fms tbw=%.1fms R2=%.3f  n_graded=%d/%d  is_box_flat=%s" %
      (out["fwhm_ms"], out["sigma_ms"], out["mu_ms"], out["tbw_ms"], out["r_squared"],
       n_graded, out["n_offsets"], is_box), flush=True)
print("WROTE %s" % args.out_json, flush=True)
