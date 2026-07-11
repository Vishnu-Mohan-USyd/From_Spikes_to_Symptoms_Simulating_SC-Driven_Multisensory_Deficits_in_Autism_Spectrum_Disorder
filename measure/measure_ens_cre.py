#!/usr/bin/env python3
"""ENSEMBLE measurement — gate 5 SBW (CRE), per seed.

Drives the EXACT validated q132_clean_cre pipeline (diag_136b_cre_5seed clean #136 CRE ruler) per seed,
loading each seed's ckpt with seed=s. SBW is spatial-canonical => SIGMA_DL_FRAMES=0.0 (the q132 regime,
distinct from gates 1-4/6 which use SIGMA_DL=3). Validated R-def = PEAK_pop. Reports the dense 2.5deg
CRE curve (negative surround lobe AS MEASURED — interpretation held pending lead relay) + per-seed
zero-cross / center-HWHM / trough. Run-only; frozen TBW/SBW md5 asserted BEFORE+AFTER.

Verified to reproduce the validated seed42 PEAK_pop CRE (peak 77.2%@0, zero-cross 29.9, HWHM 19.8,
trough -9.3%@37.5) before use on the ensemble.

Usage:  CUDA_VISIBLE_DEVICES=0 python measure_ens_cre.py [SEED ...]
"""
import os, sys, json
# ---- gate5 SBW env: spatial-canonical (q132_clean_cre) + canonical Mg-gate substrate ----
ENV = dict(TAU_GABA="10", SIGMA_DL_FRAMES="0.0", AFFERENT_SHARPEN_FACTOR="1.0", SURROUND_BUDGET="0.0",
           VAL36_BUILD="delayfix", DEND_COUPLING_ALPHA="2", MG_VHALF="-48", MG_K="0.15",
           MG_VHALF_INH="-30", GABA_SHUNT_SURR="1", K_SHUNT_SURR="0.026", E_GABA="-70.0",
           GNMDA="0.5", TAU_NMDA="40", G_REC="0.03")
for _k, _v in ENV.items():
    os.environ[_k] = _v
os.environ.pop("TAU_NMDA_V", None)
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # bundle root (this file is in <root>/measure/)
sys.path.insert(0, ROOT)
import numpy as np
import torch
from scipy.optimize import curve_fit
import routec_net_io as N
import diag_136b_cre_5seed as G
DEV = "cuda:0"

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.environ.get("OUT_JSON_OVERRIDE", os.path.join(RESULTS, "gate5_cre.json"))

DENSE_SEPS = [round(x, 1) for x in np.arange(0.0, 80.0 + 1e-6, 2.5)]
G.SEPS = DENSE_SEPS
G.NT = 60


def ckpt_for(seed):
    return N.ckpt_path_for_seed(seed)               # <root>/checkpoint/ckpt_ep79_seed{seed}_bs250_delay52_tau10_dL3.pt


def gc(x, a, s):
    return a * np.exp(-0.5 * (x / s) ** 2)


def fit_center_surround(seps, cre):
    """VERBATIM from q132_clean_cre.fit_center_surround: model-free lobe + independent center/surround fits."""
    seps = np.asarray(seps, float); cre = np.asarray(cre, float)
    peak_d, peak_v, hwhm_mf, zc = G.lobe_metrics(seps, cre)
    if zc == zc:
        zc_eff = zc
    else:
        pos = seps[cre > 0]
        zc_eff = float(pos[-1]) if pos.size else float(seps[-1])
    cmask = (seps <= zc_eff)
    cen_hwhm = cen_sigma = cen_amp = cen_r2 = float("nan")
    if cmask.sum() >= 3 and cre[0] > 0:
        try:
            p, _ = curve_fit(gc, seps[cmask], cre[cmask], p0=[max(cre[0], 1.0), 12.0],
                             bounds=([1e-3, 3.0], [np.inf, 80.0]), maxfev=20000)
            cen_amp, cen_sigma = float(p[0]), float(p[1]); cen_hwhm = 1.1774100 * cen_sigma
            resid = cre[cmask] - gc(seps[cmask], *p)
            ss = float(np.sum((cre[cmask] - cre[cmask].mean()) ** 2))
            cen_r2 = float(1.0 - np.sum(resid ** 2) / ss) if ss > 0 else float("nan")
        except Exception as e:
            print(f"   [center-fit fail]: {e}", flush=True)
    surr_min = float(np.nanmin(cre)); surr_min_d = float(seps[int(np.nanargmin(cre))])
    far_floor = float(cre[seps >= 50].mean())
    neg = seps[(cre < 0)]
    surr_last_neg = float(neg[-1]) if neg.size else float("nan")
    present = (far_floor < 0) or (zc == zc)
    return dict(peak_d=peak_d, peak_cre=peak_v, hwhm_modelfree=hwhm_mf,
                zero_cross=(None if zc != zc else float(zc)),
                cen_hwhm=cen_hwhm, cen_sigma=cen_sigma, cen_r2=cen_r2,
                surr_min=surr_min, surr_min_d=surr_min_d, far_floor=far_floor,
                surr_last_neg=(None if surr_last_neg != surr_last_neg else surr_last_neg),
                present=bool(present))


@torch.inference_mode()
def measure_seed(seed):
    ck = ckpt_for(seed)
    _mh = torch.load(ck, map_location="cpu", weights_only=False).get("mutable_hparams", {})
    net = N.load_ckpt(ck, seed, G.NT, DEV)[0]
    net.gNMDA = float(_mh.get("gNMDA", 0.5)); net.tau_nmda = float(_mh.get("tau_nmda", 40.0))
    net.g_rec = float(ENV["G_REC"])
    AV, AO, VO, locA, locV, n, S = G.capture_profiles(net)
    seps = np.array(G.SEPS, float)
    RAV_pk, RA_pk, RV_pk = AV.max(2), AO.max(2), VO.max(2)          # PEAK_pop (the validated R-def)
    B = G.cre_block(RAV_pk, RA_pk, RV_pk)
    cre_rom = np.asarray(B["cre_rom"], float)
    sh = fit_center_surround(seps, cre_rom)
    rec = {"seed": seed, "ckpt": os.path.basename(ck), "seps": seps.tolist(),
           "cre_rom": cre_rom.tolist(), **{k: sh[k] for k in
           ("peak_cre", "peak_d", "zero_cross", "cen_hwhm", "surr_min", "surr_min_d", "far_floor")}}
    print(f"[seed{seed}] peak={sh['peak_cre']:.1f}%@{sh['peak_d']:.1f}  zero-cross="
          f"{'NaN' if sh['zero_cross'] is None else round(sh['zero_cross'],1)}  HWHM="
          f"{sh['cen_hwhm']:.1f}  trough={sh['surr_min']:.1f}%@{sh['surr_min_d']:.1f}", flush=True)
    del net; torch.cuda.empty_cache()
    return rec


def _stats(vals):
    a = np.asarray([v for v in vals if v is not None], float); a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        return dict(mean=float("nan"), sd=float("nan"), sem=float("nan"), n=0)
    sd = float(np.std(a, ddof=1)) if n > 1 else 0.0
    return dict(mean=float(np.mean(a)), sd=sd, sem=(sd / np.sqrt(n) if n > 1 else 0.0), n=int(n))


def main():
    req = [int(x) for x in sys.argv[1:]] or list(range(42, 52))
    seeds = [s for s in req if os.path.exists(ckpt_for(s))]
    assert seeds, f"no ckpts for {req}"
    print(f"[ens-cre] measuring seeds {seeds}  (env: SIGMA_DL=0.0, PEAK_pop)", flush=True)
    N.assert_frozen_readouts("BEFORE")
    rows = [measure_seed(s) for s in seeds]
    N.assert_frozen_readouts("AFTER")
    seps0 = rows[0]["seps"]
    cre_mat = np.array([r["cre_rom"] for r in rows], float)  # (n_seeds, n_seps)
    agg = {"seeds": seeds, "n": len(seeds), "seps": seps0, "rows": rows, "env": ENV,
           "cre_per_offset": [_stats(cre_mat[:, j]) for j in range(cre_mat.shape[1])],
           "peak_cre": _stats([r["peak_cre"] for r in rows]),
           "zero_cross": _stats([r["zero_cross"] for r in rows]),
           "cen_hwhm": _stats([r["cen_hwhm"] for r in rows]),
           "surr_min": _stats([r["surr_min"] for r in rows]),
           "surr_min_d": _stats([r["surr_min_d"] for r in rows])}
    json.dump(agg, open(OUT, "w"), indent=1)
    print(f"\n[ens-cre] peak={agg['peak_cre']['mean']:.1f}±{agg['peak_cre']['sd']:.1f}%  "
          f"zero-cross={agg['zero_cross']['mean']:.1f}±{agg['zero_cross']['sd']:.1f}  "
          f"HWHM={agg['cen_hwhm']['mean']:.1f}±{agg['cen_hwhm']['sd']:.1f}  "
          f"trough={agg['surr_min']['mean']:.1f}±{agg['surr_min']['sd']:.1f}%@"
          f"{agg['surr_min_d']['mean']:.1f}  n={agg['n']}", flush=True)
    print(f"  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
