#!/usr/bin/env python
# task#51 route-c TBW fusion-curve probe (ep30 sanity gate + ep80 final).
# Headless. Loads 5 {prefix} ep{EP} ckpts via TBW_test.load_msi_model (restores
# asym delays from constructor_hparams + tau_nmda_inh=25 from route-c __init__,
# plasticity OFF), computes temporal_fusion P(fusion) per SOA, fits a robust
# Gaussian -> robust-FWHM (=2.355*sigma) + SDrob (=sigma), reports P(fus)@{0,+-200ms},
# peak, mu, per-model cross-seed spread. Saves npz+png.
#
# Apparatus = the CANONICAL TBW_test pipeline the debugger's route-c proof used:
#   run_fusion_across_models(fusion_method='temporal_fusion') -> P(fusion) per SOA,
#   fit_psychometric_curve(robust_fit=True) -> {fwhm, sigma, mu, tbw, r_squared}.
# Debugger route-c proof reference: robust-FWHM 396-419 / SDrob 168-178 (human 414/176).
# Degenerate "#50" signature = flat-~1.0 curve -> sigma pins near the 200ms bound.
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--epoch", type=int, required=True)
ap.add_argument("--ntrials", type=int, default=50)
ap.add_argument("--prefix", default="asymrc")
ap.add_argument("--ckpt_dir", default="/scratch/fsts_retrain_asym_20260530/checkpoints_asymdelay")
ap.add_argument("--out_dir", default="/scratch/fsts_retrain_asym_20260530/tbw")
ap.add_argument("--omin", type=int, default=-30)   # -300 ms
ap.add_argument("--omax", type=int, default=30)    # +300 ms
ap.add_argument("--ostep", type=int, default=2)    # 20 ms; hits 0 and +-20 (=+-200 ms)
ap.add_argument("--only_model", type=int, default=-1)  # >=0: run just that seed (H200 parallel)
ap.add_argument("--override_json", default="",
                help="Phase-2 inference override: JSON dict of corrected hparams "
                     "(tau_nmda_inh, tau_gaba, Erev_nmda, gNMDA, msi_inh2exc_ms). "
                     "Empty = baseline (ckpt values, true no-op).")
ap.add_argument("--config_tag", default="",
                help="label appended to output filenames (e.g. baseline, jointA, jointB).")
args = ap.parse_args()

import json
from TBW_test import run_fusion_across_models, fit_psychometric_curve

# ── Phase-2 inference override: build a modify_net hook from --override_json ──
_OVR_CFG = None
if args.override_json:
    with open(args.override_json) as _f:
        _OVR_CFG = json.load(_f)

def _make_modify_net(cfg):
    """Return a modify_net(net) that applies corrected hparams post-load and
    prints the realized live values ONCE (the override is identical across
    seeds). Returns None when cfg is empty -> baseline path is untouched."""
    if not cfg:
        return None
    from routec_overrides import apply_routec_overrides
    _state = {"printed": False}

    def _mn(net):
        applied = apply_routec_overrides(net, cfg)
        if not _state["printed"]:
            print(f"[override] applied={ {k: v for k, v in applied.items() if k != 'live'} }", flush=True)
            print(f"[override] LIVE   ={applied.get('live')}", flush=True)
            _state["printed"] = True
    return _mn

_MODIFY_NET = _make_modify_net(_OVR_CFG)

CK = Path(args.ckpt_dir)
model_ids = [args.only_model] if args.only_model >= 0 else list(range(5))
paths = [CK / f"{args.prefix}_m{i}_ep{args.epoch:02d}.pt" for i in model_ids]
missing = [str(p) for p in paths if not p.exists()]
if missing:
    print("MISSING_CKPTS:", missing, flush=True)
    sys.exit(3)

offsets = list(range(args.omin, args.omax + 1, args.ostep))
print(f"[tbw] prefix={args.prefix} epoch={args.epoch} ntrials={args.ntrials} "
      f"n_offsets={len(offsets)} SOA {args.omin*10}..{args.omax*10}ms step {args.ostep*10}ms",
      flush=True)

pooled = run_fusion_across_models(
    paths, offsets, device="cuda",
    fusion_method="temporal_fusion", n_trials=args.ntrials,
    modify_net=_MODIFY_NET, collect_npeaks=True,
)
oms = np.array(pooled["offsets_ms"], dtype=float)
mf = np.array(pooled["mean_fusion"], dtype=float)
sem = np.array(pooled["sem_fusion"], dtype=float)
allf = np.array(pooled["all_fusion"], dtype=float)  # (n_models, n_offsets)
mnpk = np.array(pooled["mean_npeaks"], dtype=float)  # (n_offsets,) mean #peaks/trial
allnpk = np.array(pooled["all_npeaks"], dtype=float)  # (n_models, n_offsets)


def at(target_ms):
    j = int(np.argmin(np.abs(oms - target_ms)))
    return oms[j], mf[j], sem[j]


print("\n== POOLED P(fusion) ==", flush=True)
for t in (0, -200, 200):
    o, p, s = at(t)
    print(f"  SOA {t:+5d}ms (offset {o:+.0f}): P(fus)={p:.3f} +/- {s:.3f}")
pk = int(np.argmax(mf))
print(f"  peak P(fus)={mf[pk]:.3f} at SOA {oms[pk]:+.0f}ms")

# pooled robust Gaussian fit (robust_fit=True default) -> the debugger's FWHM/SDrob
fit = fit_psychometric_curve(oms, mf)
print("\n== ROBUST GAUSSIAN FIT (pooled) ==", flush=True)
print(f"  robust-FWHM={fit['fwhm']:.1f}ms  SDrob(sigma)={fit['sigma']:.1f}ms  "
      f"mu={fit['mu']:.1f}ms  tbw(crit)={fit['tbw']:.1f}ms  R2={fit['r_squared']:.3f}")
print("  (human target FWHM~414 / SDrob~176 ; #50-degenerate=flat-1.0 -> sigma~200 bound)")

# per-model robust fits -> cross-seed spread (the '5/5 ckpts FWHM lo-hi' framing)
print("\n== PER-MODEL robust fit ==", flush=True)
j0 = int(np.argmin(np.abs(oms)))
fwhms, sigs, mus = [], [], []
for k in range(allf.shape[0]):
    mid = model_ids[k]
    try:
        fi = fit_psychometric_curve(oms, allf[k])
        fwhms.append(fi["fwhm"]); sigs.append(fi["sigma"]); mus.append(fi["mu"])
        # count strictly-intermediate P values (0<P<1) -> graded(bell) vs binary(box) discriminator
        ng = int(np.sum((allf[k] > 0.02) & (allf[k] < 0.98)))
        print(f"  m{mid}: FWHM={fi['fwhm']:.1f} SDrob={fi['sigma']:.1f} mu={fi['mu']:.1f} "
              f"R2={fi['r_squared']:.3f} P(fus@0)={allf[k][j0]:.3f} peak={allf[k].max():.3f} n_graded={ng}/{allf.shape[1]}")
    except Exception as e:
        print(f"  m{mid}: FIT_FAIL {e!r}")
if fwhms:
    print(f"  -> FWHM {min(fwhms):.0f}-{max(fwhms):.0f} (mean {np.mean(fwhms):.0f}) ; "
          f"SDrob {min(sigs):.0f}-{max(sigs):.0f} (mean {np.mean(sigs):.0f})")

print("\n== FULL CURVE (SOA_ms:P) ==", flush=True)
print("  " + "  ".join(f"{int(o):+d}:{p:.2f}" for o, p in zip(oms, mf)))

# Per-SOA MSI peak-count dump (Phase-2): directly exposes the is_temporally_fused
# (<=1 peak => fused) behaviour. A box (P=1 everywhere) with mean_npeaks~1 across
# all SOA = single sustained MSI peak. If corrected timing splits the response,
# mean_npeaks rises toward 2 at large |SOA| -> graded fusion territory.
print("\n== PER-SOA MEAN #PEAKS/trial (SOA_ms:npk) ==", flush=True)
print("  " + "  ".join(f"{int(o):+d}:{v:.2f}" for o, v in zip(oms, mnpk)))
_j0 = int(np.argmin(np.abs(oms)))
_jlo = int(np.argmin(np.abs(oms - (-200)))); _jhi = int(np.argmin(np.abs(oms - 200)))
print(f"  npeaks @SOA 0={mnpk[_j0]:.2f}  -200ms={mnpk[_jlo]:.2f}  +200ms={mnpk[_jhi]:.2f}  "
      f"(min={mnpk.min():.2f} max={mnpk.max():.2f})", flush=True)

# persist
out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
_tag = (f"_{args.config_tag}" if args.config_tag else "")
stem = (f"{args.prefix}_tbw_ep{args.epoch:02d}{_tag}"
        + (f"_m{args.only_model}" if args.only_model >= 0 else ""))
np.savez(out / f"{stem}.npz",
         offsets_ms=oms, mean_fusion=mf, sem_fusion=sem, all_fusion=allf,
         mean_npeaks=mnpk, all_npeaks=allnpk,
         fwhm=fit["fwhm"], sigma=fit["sigma"], mu=fit["mu"], tbw=fit["tbw"],
         r_squared=fit["r_squared"],
         config_tag=args.config_tag, override_json=args.override_json)
plt.figure(figsize=(7, 4))
plt.errorbar(oms, mf, yerr=sem, fmt="o", ms=3, capsize=2, label="P(fus) pooled +- SEM")
plt.plot(fit["xs"], fit["ys"], "-", lw=1.5, label=f"robust Gauss FWHM={fit['fwhm']:.0f}ms")
plt.axvline(0, color="k", lw=0.5, ls=":")
plt.ylim(-0.02, 1.05)
plt.xlabel("SOA (ms; - = V-leading per #46 asym convention)")
plt.ylabel("P(fusion)")
plt.title(f"{args.prefix} ep{args.epoch} TBW (temporal_fusion, ntrials={args.ntrials})")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(out / f"{stem}.png", dpi=110)
print(f"\nSAVED {out}/{stem}.npz + {stem}.png", flush=True)
print("TBW_PROBE_DONE", flush=True)
