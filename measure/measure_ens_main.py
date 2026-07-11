#!/usr/bin/env python3
"""ENSEMBLE measurement — gates 1(RATE) 2(TBW) 3(E/I shunt-aware) 4(MEI), per seed.

Drives the EXACT validated measurement FUNCTIONS (eitbw_gnmda050 + q5_53_scorecard lineage) per seed,
loading each seed's ckpt with seed=s (honest per-seed measurement noise — the grade scripts hardcode
seed42 because they only measured seed42). Full canonical substrate env (= grade_vadvance.sh
SUBSTRATE + FORMA_COMMON + EXP_G_REC). load_ckpt restores the as-trained operating point (sigma_dL=3,
gNMDA=0.5, tau_nmda=40, g_rec=0.03) from the ckpt mutable_hparams, so each seed is measured as trained.

RUN-ONLY on the frozen TBW/SBW readouts (md5 asserted BEFORE+AFTER). Writes per-seed rows + per-gate
mean/SD/SEM. Verified to reproduce the validated seed42 scorecard (rate 17.19, E/I 1.059, TBW ~260,
MEI 4.00/7.60/6.15/1.77/0.97/0.99) before use on the ensemble.

Usage:  CUDA_VISIBLE_DEVICES=0 python measure_ens_main.py [SEED ...]   (default: all available)
"""
import os, sys, json, glob
# ---- CANONICAL substrate env (grade_vadvance.sh L16-17 SUBSTRATE+FORMA_COMMON+EXP_G_REC), BEFORE imports ----
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
import measure_107_convergence as M
import q5_53_scorecard as S
import inverse_effectiveness_routec as IE
MDC, V, G88, NT = M.MDC, M.V, M.G88, M.NT
DEV = "cuda:0"

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.environ.get("OUT_JSON_OVERRIDE", os.path.join(RESULTS, "gates_main.json"))


def ckpt_for(seed):
    return N.ckpt_path_for_seed(seed)               # <root>/checkpoint/ckpt_ep79_seed{seed}_bs250_delay52_tau10_dL3.pt


def available_seeds(req):
    seeds = []
    for s in req:
        if os.path.exists(ckpt_for(s)):
            seeds.append(s)
    return seeds


@torch.no_grad()
def measure_seed(seed):
    ck = ckpt_for(seed)
    rec = {"seed": seed, "ckpt": os.path.basename(ck)}

    # ---- gate4 MEI (validator-pristine: run before the additive shunt patch) ----
    net = N.load_ckpt(ck, seed, 1, DEV)[0]
    mei, RA, RB = S.measure_ie_row(net)
    rec["mei"] = [float(x) for x in mei]
    rec["ie_intensities"] = list(IE.INTENSITIES)
    rec["mei_shape"] = S.shape(mei)
    del net; torch.cuda.empty_cache()

    # ---- gate1 RATE: bimodal MSI window-mean + per-frame PSTH (sigma_in=5.0, the as-graded value) ----
    net = N.load_ckpt(ck, seed, NT, DEV)[0]
    sc = V.T.run_sc_diagnostics(net, modality="B", sigma_in=5.0, verbose=False)
    rec["msi_hz"] = float(sc["spike_rates"]["MSI"])
    _spk = np.asarray(sc["raw_time_series"]["MSI_spikes"], float)   # per-frame total MSI spikes
    _frame_ms = int(net.n_substeps) * float(net.dt)
    rec["psth_hz"] = (_spk / int(net.n) / (_frame_ms / 1000.0)).tolist()  # per-frame population rate (Hz)
    rec["frame_ms"] = _frame_ms
    del net; torch.cuda.empty_cache()

    # ---- gate2 TBW: raw half-max width of the P(fusion) psychometric (sigma_dL=3.0 readout) ----
    net = N.load_ckpt(ck, seed, NT, DEV)[0]
    o, f, n, eff = MDC.tbw_fused_counts_jit(net, 3.0, run_seed=0, nt=NT)
    offs = np.asarray(o, float); pf = np.asarray(f, float) / np.asarray(n, float)
    asc = np.argsort(offs); offs_s, pf_s = offs[asc], pf[asc]
    pk = float(np.nanmax(pf)); half = 0.5 * pk
    above = offs[pf >= half]
    raw_fwhm = float(np.nanmax(above) - np.nanmin(above)) if above.size >= 2 else float("nan")
    rec["tbw_raw_fwhm"] = raw_fwhm
    rec["tbw_peak"] = pk
    rec["tbw_offsets"] = offs_s.tolist()
    rec["tbw_pfusion"] = pf_s.tolist()
    del net; torch.cuda.empty_cache()

    # ---- gate3 E/I shunt-aware: averaged (E,I) point at SYNC (offset 0) + sync ratio ----
    net = N.load_ckpt(ck, seed, 1, DEV)[0]
    S.patch_store_shunt(net)
    ei = S.measure_ei_shuntaware(net)
    sig = float(getattr(net, "sigma_in", 10.0))
    E, Is, Ish = S.ei_probe(net, 0, sig, pulse_frames=5, n_frames=20, intensity=1.0)  # sync
    rec["ei_sync"] = float(ei["shunt_sync"])
    rec["ei_off"] = float(ei["shunt_off"])
    rec["E_sync"] = float(np.mean(E))          # network-averaged excitatory current at sync
    rec["I_sync"] = float(np.mean(Ish))        # network-averaged shunt-aware inhibitory current at sync
    rec["E_off"] = float(ei["exc_off"])
    rec["I_off"] = float(ei["inh_shunt_off"])
    del net; torch.cuda.empty_cache()

    print(f"[seed{seed}] rate={rec['msi_hz']:.3f}Hz  TBW={rec['tbw_raw_fwhm']:.0f}ms  "
          f"E/I_sync={rec['ei_sync']:.3f}  (E,I)_sync=({rec['E_sync']:.3f},{rec['I_sync']:.3f})  "
          f"MEI={' '.join(f'{x:.2f}' for x in rec['mei'])} {rec['mei_shape']}", flush=True)
    return rec


def _stats(vals):
    a = np.asarray(vals, float); a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        return dict(mean=float("nan"), sd=float("nan"), sem=float("nan"), n=0)
    sd = float(np.std(a, ddof=1)) if n > 1 else 0.0
    return dict(mean=float(np.mean(a)), sd=sd, sem=(sd / np.sqrt(n) if n > 1 else 0.0), n=int(n))


def main():
    req = [int(x) for x in sys.argv[1:]] or list(range(42, 52))
    seeds = available_seeds(req)
    assert seeds, f"no ckpts found for {req}"
    print(f"[ens-main] measuring seeds {seeds}  (env: substrate+FORMA, SIGMA_DL=3)", flush=True)
    N.assert_frozen_readouts("BEFORE")
    rows = [measure_seed(s) for s in seeds]
    N.assert_frozen_readouts("AFTER")

    agg = {"seeds": seeds, "n": len(seeds), "rows": rows, "env": ENV}
    agg["rate_hz"] = _stats([r["msi_hz"] for r in rows])
    agg["tbw_raw_fwhm"] = _stats([r["tbw_raw_fwhm"] for r in rows])
    agg["ei_sync"] = _stats([r["ei_sync"] for r in rows])
    agg["E_sync"] = _stats([r["E_sync"] for r in rows])
    agg["I_sync"] = _stats([r["I_sync"] for r in rows])
    # per-intensity MEI mean/SD across seeds
    mei_mat = np.array([r["mei"] for r in rows], float)  # (n_seeds, 6)
    agg["mei_per_intensity"] = [_stats(mei_mat[:, j]) for j in range(mei_mat.shape[1])]
    agg["ie_intensities"] = list(IE.INTENSITIES)
    # TBW pfusion curves (per-SOA mean/SD) — only if all share the same offset grid
    offs0 = rows[0]["tbw_offsets"]
    if all(r["tbw_offsets"] == offs0 for r in rows):
        pf_mat = np.array([r["tbw_pfusion"] for r in rows], float)
        agg["tbw_offsets"] = offs0
        agg["tbw_pfusion_per_offset"] = [_stats(pf_mat[:, j]) for j in range(pf_mat.shape[1])]
    # gate1 PSTH (per-frame mean/SD across seeds) — only if all share the same frame count
    if len({len(r["psth_hz"]) for r in rows}) == 1:
        psth_mat = np.array([r["psth_hz"] for r in rows], float)
        agg["frame_ms"] = rows[0]["frame_ms"]
        agg["psth_per_frame"] = [_stats(psth_mat[:, j]) for j in range(psth_mat.shape[1])]
    json.dump(agg, open(OUT, "w"), indent=1)
    print(f"\n[ens-main] rate={agg['rate_hz']['mean']:.2f}±{agg['rate_hz']['sd']:.2f}  "
          f"TBW={agg['tbw_raw_fwhm']['mean']:.0f}±{agg['tbw_raw_fwhm']['sd']:.0f}  "
          f"E/I={agg['ei_sync']['mean']:.3f}±{agg['ei_sync']['sd']:.3f}  "
          f"(E,I)=({agg['E_sync']['mean']:.3f}±{agg['E_sync']['sd']:.3f},"
          f"{agg['I_sync']['mean']:.3f}±{agg['I_sync']['sd']:.3f})  n={agg['n']}", flush=True)
    print(f"[ens-main] MEI/intensity: " +
          "  ".join(f"{IE.INTENSITIES[j]}:{agg['mei_per_intensity'][j]['mean']:.2f}±{agg['mei_per_intensity'][j]['sd']:.2f}"
                    for j in range(len(IE.INTENSITIES))), flush=True)
    print(f"  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
