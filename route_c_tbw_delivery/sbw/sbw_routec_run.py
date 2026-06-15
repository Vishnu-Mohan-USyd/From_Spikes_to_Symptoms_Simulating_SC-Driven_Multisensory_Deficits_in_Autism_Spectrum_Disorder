#!/usr/bin/env python
"""sbw_routec_run.py — measure SBW on ONE ep80 checkpoint with the VALIDATED
SBW_test.py apparatus (md5 73b7d136), invoked IDENTICALLY to make_curves.make_sbw().

ONE ckpt per process so the 5-seed ensemble runs fully in PARALLEL (Rule 6).
compute_sbw_fused_persep uses an UNSEEDED np.random.default_rng() with NO
cross-ckpt state, so per-ckpt parallelization is semantically identical to
make_curves' sequential loop (each seed is an independent Monte-Carlo estimate).

Apparatus settings (frozen, == make_curves.make_sbw):
  seps   = range(-80, 85, 5)         # 33 pts, #39 grid
  method = compute_sbw_fused_persep  # is_fused P(fusion), plasticity-OFF
  n_trials=50, intensity=0.5, duration=20, block_size=32 (default), linear=False
  half-width = abs(fit_pedestal_curve(...)[2])   # popt[2] = w  (== make_curves hw)
"""
import os, sys, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--eval_app", default="/scratch/eval_apparatus")
ap.add_argument("--code_dir", required=True, help="dir whose Training.py defines the net for THIS lineage")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tag", required=True, help="e.g. asymrc_m0")
ap.add_argument("--out_json", required=True)
ap.add_argument("--override_json", default="",
                help="Phase-2 inference override: JSON dict of corrected hparams "
                     "(tau_nmda_inh, tau_gaba, Erev_nmda, gNMDA, msi_inh2exc_ms). "
                     "Empty = baseline (ckpt values, true no-op).")
args = ap.parse_args()

sys.path.insert(0, args.eval_app)
sys.path.insert(0, args.code_dir)   # so SBW_test's `from Training import *` -> THIS lineage
import torch
# torch>=2.6 defaults weights_only=True, which refuses these trusted research
# ckpts (constructor_hparams dict). Restore pre-2.6 load behavior — changes ONLY
# the unpickling safety flag, not the weights or the metric. Apparatus untouched.
_orig_torch_load = torch.load
def _torch_load_compat(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _torch_load_compat
import SBW_test as S

device = "cuda" if torch.cuda.is_available() else "cpu"
seps = list(range(-80, 85, 5))
print("=== %s === code_dir=%s SBW_test=%s device=%s torch=%s" %
      (args.tag, args.code_dir, getattr(S, "__file__", "?"), device, torch.__version__), flush=True)

net = S.load_msi_model(args.ckpt, device=device)

# ── Phase-2 inference override (post-load setattr; apparatus + Training.py untouched) ──
override_applied = None
if args.override_json:
    with open(args.override_json) as _f:
        _ovr_cfg = json.load(_f)
    from routec_overrides import apply_routec_overrides  # resolves via --code_dir on sys.path
    override_applied = apply_routec_overrides(net, _ovr_cfg)
    print("[OVERRIDE] %s applied=%s" % (
        args.tag, {k: v for k, v in override_applied.items() if k != "live"}), flush=True)
    print("[OVERRIDE] %s LIVE=%s" % (args.tag, override_applied.get("live")), flush=True)

wiring = dict(ckpt=os.path.basename(args.ckpt),
              tau_gaba=float(getattr(net, "tau_gaba", float("nan"))),
              Erev_nmda=float(getattr(net, "Erev_nmda", float("nan"))),
              gNMDA=float(getattr(net, "gNMDA", float("nan"))),
              msi_inh2exc_ms=float(getattr(net, "conduction_delay_msi_inh2exc_ms", float("nan"))),
              tau_nmda_inh=float(getattr(net, "tau_nmda_inh", float("nan"))),
              tau_nmda=float(getattr(net, "tau_nmda", float("nan"))),
              d_a2msi=int(getattr(net, "conduction_delay_a2msi", -1)),
              d_v2msi=int(getattr(net, "conduction_delay_v2msi", -1)),
              sigma_in=float(getattr(net, "sigma_in", float("nan"))),
              n=int(getattr(net, "n", -1)), space_size=float(getattr(net, "space_size", float("nan"))),
              plasticity_enabled=bool(getattr(net, "plasticity_enabled", True)))
print("[WIRING] %s %s" % (args.tag, wiring), flush=True)

pf = np.asarray(S.compute_sbw_fused_persep(
    net, separations_deg=seps, n_trials=50, intensity=0.5, duration=20), float)
try:
    _, _, popt = S.fit_pedestal_curve({"separations_deg": np.array(seps, float), "mean_prob": pf})
    hw = float(abs(popt[2]))
except Exception as e:
    hw = float("nan"); print("[SBW] %s fit fail: %r" % (args.tag, e), flush=True)
ev = S.compute_sbw_enhancement_persep(
    net, separations_deg=seps, n_trials=50, intensity=0.5, duration=20, roi_half=20)
me = ev[0] if isinstance(ev, (tuple, list)) else ev

out = dict(tag=args.tag, ckpt=args.ckpt, wiring=wiring, separations_deg=seps,
           n_trials=50, intensity=0.5, duration=20, roi_half=20,
           override_json=args.override_json, override=override_applied,
           apparatus="SBW_test.compute_sbw_fused_persep (is_fused P(fusion), plasticity-OFF); hw=|fit_pedestal_curve popt[2]|",
           pfusion=pf.tolist(), halfwidth_deg=hw, enhancement=np.asarray(me, float).tolist())
json.dump(out, open(args.out_json, "w"), indent=1)
print("[SBW] %s hw=%.2f Pf=%.2f..%.2f -> %s" % (args.tag, hw, pf.min(), pf.max(), args.out_json), flush=True)
