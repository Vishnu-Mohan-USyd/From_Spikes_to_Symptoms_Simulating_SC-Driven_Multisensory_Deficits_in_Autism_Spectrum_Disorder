#!/usr/bin/env python3
"""ENSEMBLE gate6 (NEW) — multisensory latency BENEFIT vs stimulus intensity (inverse effectiveness).

USER CHANGE (relayed by lead): show the latency benefit at LOW intensity, on the inverse-effectiveness
principle (multisensory benefits are largest for weak stimuli). REPLACES the single full-intensity
A/V/B bars (which sit at the race-model bound = honest null).

Per seed, measure first-spike latency L_A, L_V, L_B (bimodal, simultaneous onset) across the gate4/IE
intensity grid [0.05 .. 1.6]; compute the RACE-MODEL benefit = min(L_A, L_V) - L_B per intensity
(positive = bimodal first spike FASTER than the fastest unimodal = a true MSI latency facilitation).
Aggregate mean/SD/SEM across the ensemble. Inverse effectiveness predicts the benefit is LARGEST at LOW
intensity; if it is NULL even at low intensity, that is reported AS NULL (never cherry-picked).

Faithful reuse: drives the VALIDATED response_latency_routec.measure_latency / first_spike_latency
(substep first-spike probe), varying only the module-global INTENSITY (the Gaussian pulse amplitude).
A dedicated I=1.0 anchor on seed42 must reproduce the validated full-intensity gate6 (L_A=33.7, L_V=48.6,
L_B=33.7) -> proves the intensity-reuse is faithful. Run-only on frozen TBW/SBW md5 (BEFORE==AFTER).

Usage:  CUDA_VISIBLE_DEVICES=0 python measure_ens_latency_sweep.py [SEED ...]
"""
import os, sys, json
# ---- canonical substrate env (= gates 1-4/6 operating point, SIGMA_DL=3), BEFORE imports ----
ENV = dict(DEND_COUPLING_ALPHA="2", MG_VHALF="-48", MG_VHALF_INH="-30", MG_K="0.15",
           GABA_SHUNT_SURR="1", K_SHUNT_SURR="0.026", E_GABA="-70.0", TAU_GABA="10",
           SIGMA_DL_FRAMES="3", GNMDA="0.50", TAU_NMDA="40", G_REC="0.03", EXP_G_REC="0.03",
           K_DVDT="0.0", TAU_DVDT="3.0", V_THRESH_FLOOR="20.0", DVDT_CAP="50.0")
for _k, _v in ENV.items():
    os.environ[_k] = _v
os.environ.pop("TAU_NMDA_V", None)
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # bundle root (this file is in <root>/measure/)
sys.path.insert(0, ROOT)
import numpy as np
import torch
import routec_net_io as N
import response_latency_routec as RL
import inverse_effectiveness_routec as IE
DEV = "cuda:0"

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.environ.get("OUT_JSON_OVERRIDE", os.path.join(RESULTS, "gate6_latency_sweep.json"))
# IE/gate4 grid (lead: SAME grid) + 1.0 = the validated full-intensity anchor point
INTENSITIES = sorted(set(float(x) for x in IE.INTENSITIES) | {1.0})   # [0.05,0.1,0.2,0.4,0.8,1.0,1.6]
N_TRIALS = 128
ANCHOR_I = 1.0
ANCHOR_REF = (33.7, 48.6, 33.7)   # validated seed42 full-intensity gate6 (L_A, L_V, L_B)


def ckpt_for(seed):
    return N.ckpt_path_for_seed(seed)               # <root>/checkpoint/ckpt_ep79_seed{seed}_bs250_delay52_tau10_dL3.pt


@torch.no_grad()
def measure_seed(seed):
    ck = ckpt_for(seed)
    net = N.load_ckpt(ck, seed, N_TRIALS, DEV)[0]
    sigma_in = float(net.sigma_in)
    LA, LV, LB, BEN, NA, NV, NB = [], [], [], [], [], [], []
    _saved = RL.INTENSITY
    try:
        for I in INTENSITIES:
            RL.INTENSITY = float(I)                                  # vary pulse amplitude (faithful reuse)
            m = RL.measure_latency(net, sigma_in, N_TRIALS, DEV)     # simultaneous onset (soa=0)
            la, lv, lb = m["L_A_ms"], m["L_V_ms"], m["L_B_ms"]
            uni_min = float(np.nanmin([la, lv])) if not (np.isnan(la) and np.isnan(lv)) else float("nan")
            ben = (uni_min - lb) if (uni_min == uni_min and lb == lb) else float("nan")
            LA.append(la); LV.append(lv); LB.append(lb); BEN.append(ben)
            NA.append(m["per_modality"]["A"]["n_spiked"]); NV.append(m["per_modality"]["V"]["n_spiked"])
            NB.append(m["per_modality"]["B"]["n_spiked"])
    finally:
        RL.INTENSITY = _saved
    rec = {"seed": seed, "ckpt": os.path.basename(ck), "intensities": INTENSITIES, "sigma_in": sigma_in,
           "L_A": LA, "L_V": LV, "L_B": LB, "benefit": BEN,
           "n_spiked_A": NA, "n_spiked_V": NV, "n_spiked_B": NB, "n_trials": N_TRIALS}
    # ---- seed42 faithfulness anchor at I=1.0 ----
    if seed == 42:
        j = INTENSITIES.index(ANCHOR_I)
        got = (LA[j], LV[j], LB[j])
        ok = all(abs(g - r) <= 2.0 for g, r in zip(got, ANCHOR_REF))
        rec["anchor_I1.0"] = dict(got=got, ref=ANCHOR_REF, pass_=bool(ok))
        flag = "OK" if ok else "*** ANCHOR MISMATCH ***"
        print(f"[seed42 ANCHOR I=1.0] L_A/L_V/L_B = {got[0]:.1f}/{got[1]:.1f}/{got[2]:.1f}  "
              f"ref {ANCHOR_REF}  -> {flag}", flush=True)
    bs = "  ".join(f"{I:g}:{('nan' if b!=b else f'{b:+.1f}')}" for I, b in zip(INTENSITIES, BEN))
    print(f"[seed{seed}] benefit min(A,V)-B (ms) vs intensity:  {bs}", flush=True)
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
    print(f"[ens-lat-sweep] measuring seeds {seeds}  intensities {INTENSITIES}  (SIGMA_DL=3, soa=0)", flush=True)
    N.assert_frozen_readouts("BEFORE")
    rows = [measure_seed(s) for s in seeds]
    N.assert_frozen_readouts("AFTER")
    nI = len(INTENSITIES)
    ben = np.array([r["benefit"] for r in rows], float)         # (n_seeds, nI)
    LA = np.array([r["L_A"] for r in rows], float)
    LV = np.array([r["L_V"] for r in rows], float)
    LB = np.array([r["L_B"] for r in rows], float)
    agg = {"seeds": seeds, "n": len(seeds), "intensities": INTENSITIES, "rows": rows, "env": ENV,
           "benefit_per_intensity": [_stats(ben[:, j]) for j in range(nI)],
           "L_A_per_intensity": [_stats(LA[:, j]) for j in range(nI)],
           "L_V_per_intensity": [_stats(LV[:, j]) for j in range(nI)],
           "L_B_per_intensity": [_stats(LB[:, j]) for j in range(nI)]}
    json.dump(agg, open(OUT, "w"), indent=1)
    bmeans = [agg["benefit_per_intensity"][j]["mean"] for j in range(nI)]
    bsds = [agg["benefit_per_intensity"][j]["sd"] for j in range(nI)]
    print("\n[ens-lat-sweep] benefit min(A,V)-B (ms), mean±SD vs intensity:", flush=True)
    for j, I in enumerate(INTENSITIES):
        print(f"    I={I:<5g}  benefit={bmeans[j]:+.2f}±{bsds[j]:.2f} ms   "
              f"L_A={agg['L_A_per_intensity'][j]['mean']:.1f}  L_V={agg['L_V_per_intensity'][j]['mean']:.1f}  "
              f"L_B={agg['L_B_per_intensity'][j]['mean']:.1f}", flush=True)
    jlo = int(np.nanargmin(INTENSITIES))
    jmax = int(np.nanargmax(bmeans))
    print(f"  LOW-intensity (I={INTENSITIES[jlo]:g}) benefit = {bmeans[jlo]:+.2f}±{bsds[jlo]:.2f} ms ;  "
          f"MAX benefit = {bmeans[jmax]:+.2f} ms @ I={INTENSITIES[jmax]:g} ;  "
          f"HIGH (I={INTENSITIES[-1]:g}) = {bmeans[-1]:+.2f} ms", flush=True)
    print(f"  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
