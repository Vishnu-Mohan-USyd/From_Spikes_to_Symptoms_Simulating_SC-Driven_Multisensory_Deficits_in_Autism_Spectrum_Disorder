#!/usr/bin/env python3
"""DEBUGGER #397 — ONE dose-response point: measure TBW raw-FWHM on FROZEN dm10 weights under a
SINGLE-VARIABLE inference-time lesion of an existing mechanism. NO retrain, NO weight edit.

Reuses the VALIDATED drace_leverD load+TBW path VERBATIM (BASE_ENV = dm10 point; MDC.tbw_fused_counts_jit
= frozen readout md5 80d33465). The ONLY thing that varies across points is one of {aM,dM | pv_gaba | tau_gaba | gNMDA},
each a LIVE net scalar (aM/dM L3260/3266; tau_gaba L2856; gNMDA L3071; pv_gaba_scale L3205) overridden AFTER load.
All overrides are scalars NOT in the state_dict => weights stay bit-identical (asserted). Firewall md5 BEFORE==AFTER.

Usage: CUDA_VISIBLE_DEVICES=g python3 tbw_point.py --ckpt PATH --tag LABEL \
         [--aM 0.02] [--dM 10] [--pv_gaba 1.0] [--tau_gaba 18] [--gnmda 0.51] [--meas_seed 44] --out JSON
Defaults = the dm10 operating point. Pass only the knob you are perturbing.
"""
import os, sys, hashlib, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--aM", type=float, default=0.02, help="exc adaptation recovery rate (dm10=0.02; OFF=0.0; baseline=0.008)")
ap.add_argument("--dM", type=float, default=10.0, help="exc adaptation spike increment (dm10=10; OFF=0; baseline=8)")
ap.add_argument("--pv_gaba", type=float, default=1.0, help="disynaptic GABA conductance scale pv_gaba_scale (dm10=1.0=x5.56; OFF=0)")
ap.add_argument("--tau_gaba", type=float, default=18.0, help="GABA_A decay ms (dm10=18)")
ap.add_argument("--gnmda", type=float, default=0.51, help="global NMDA conductance gNMDA (dm10=0.51; OFF=0)")
ap.add_argument("--meas_seed", type=int, default=44)
ap.add_argument("--out", default="")
ap.add_argument("--device", default="cuda:0")
A = ap.parse_args()

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECT = {"TBW_test.py": "80d33465c4bf55d6e85b5990acb92da7", "SBW_test.py": "73b7d13626964d851cc090818b728311"}
MEAS_SEED = int(A.meas_seed)

# VERBATIM dm10 BASE_ENV (from gate6_dm10.sh / drace_leverD). tau_gaba/gnmda/u_stp overridden on net after load.
# NOTE: load_ckpt (val36 L128/L148) asserts the ENV TAU_GABA/GNMDA == the ckpt's TRAINED value ("measurement
# mirrors training"). So the ENV must stay at dm10's trained regime (tau_gaba=18, gNMDA=0.51) to pass the load
# guard; the tau_gaba/gNMDA LESION is applied ONLY as a post-load live-scalar override (net.tau_gaba/net.gNMDA below).
BASE_ENV = dict(DEND_COUPLING_ALPHA="2", MG_VHALF="-48", MG_VHALF_INH="-30", MG_K="0.15",
                GABA_SHUNT_SURR="1", K_SHUNT_SURR="0.026", E_GABA="-70.0", TAU_GABA="18.0", G_GABA="5.56",
                SIGMA_DL_FRAMES="3", GNMDA="0.51", TAU_NMDA="40", G_REC="0.03", EXP_G_REC="0.03",
                K_DVDT="0.0", TAU_DVDT="3.0", V_THRESH_FLOOR="20.0", DVDT_CAP="50.0",
                AFFERENT_JITTER_MS="4", MPLBACKEND="Agg",
                U_STP_A="0.2", U_STP_V="0.2", NMDA_STD_SCALE="0.8", TAU_REC="400.0")
for k, v in BASE_ENV.items():
    os.environ[k] = v
for k in ("TAU_NMDA_V", "ADAPT_A", "ADAPT_DM", "ADAPT_A_INH", "ADAPT_DM_INH"):
    os.environ.pop(k, None)          # adaptation set on the net directly (aM/dM), not via env
os.environ["SEED"] = str(MEAS_SEED)

sys.path.insert(0, BUNDLE)
import torch                                            # noqa: E402
import routec_net_io as N                               # noqa: E402
import measure_107_convergence as M                     # noqa: E402  (imports MDC; sets TAU_GABA=10 in env -> we override net.tau_gaba after load)
MDC, V, NT = M.MDC, M.V, M.NT
DEV = A.device


def fw(tag):
    for lbl, p, exp in [("TBW", N.TBW_PATH, EXPECT["TBW_test.py"]), ("SBW", N.SBW_PATH, EXPECT["SBW_test.py"])]:
        got = hashlib.md5(open(p, "rb").read()).hexdigest()
        assert got == exp, f"FIREWALL {tag} {lbl} {got}!={exp}"
    N.assert_frozen_readouts(tag)
    return EXPECT["TBW_test.py"], EXPECT["SBW_test.py"]


def wfp(net):
    """bit-identity fingerprint over the weight state_dict (scalars aM/dM/pv/tau/gNMDA are NOT params -> unaffected)."""
    h = hashlib.md5()
    for k in sorted(net.state_dict().keys()):
        t = net.state_dict()[k]
        if torch.is_tensor(t):
            h.update(k.encode()); h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    md5 = hashlib.md5(open(A.ckpt, "rb").read()).hexdigest()
    print(f"[397 {A.tag}] ckpt md5={md5} meas_seed={MEAS_SEED} dev={DEV}", flush=True)
    fwb = fw("BEFORE")

    os.environ["SIGMA_DL_FRAMES"] = "3"
    net, res, ep, grec_env, cm = N.load_ckpt(A.ckpt, MEAS_SEED, NT, DEV)
    # ---- SINGLE-VARIABLE overrides on the FROZEN net (live scalars; NOT weights) ----
    net.g_rec = 0.03
    net.aM = float(A.aM); net.dM = float(A.dM)                # adaptation lesion
    net.pv_gaba_scale = float(A.pv_gaba)                      # GABA conductance scale
    net.tau_gaba = float(A.tau_gaba)                          # GABA decay
    net.gNMDA = float(A.gnmda)                                # NMDA conductance
    net.u_stp_a = 0.2; net.u_stp_v = 0.2; net.nmda_std_scale = 0.8
    net.u_a.fill_(0.2); net.u_v.fill_(0.2)
    net.plasticity_enabled = False

    fp_before = wfp(net)
    eff = dict(aM=float(net.aM), dM=float(net.dM), pv_gaba_scale=float(net.pv_gaba_scale),
               tau_gaba=float(net.tau_gaba), gNMDA=float(net.gNMDA),
               u_stp_a=float(net.u_stp_a), nmda_std_scale=float(net.nmda_std_scale),
               g_rec=float(net.g_rec), epoch=int(ep))
    print(f"  EFFECTIVE aM={eff['aM']:.4f} dM={eff['dM']:.1f} pv_gaba={eff['pv_gaba_scale']:.3f} "
          f"tau_gaba={eff['tau_gaba']:.1f} gNMDA={eff['gNMDA']:.3f} u_stp={eff['u_stp_a']:.2f} "
          f"nmda_std={eff['nmda_std_scale']:.2f} g_rec={eff['g_rec']:.3f} ep{ep}", flush=True)

    # ---- TBW raw-FWHM, 3 run-seeds (VERBATIM drace_leverD formula) ----
    tbws, pks = [], []
    for rs in (0, 1, 2):
        o, f, n, _ = MDC.tbw_fused_counts_jit(net, 3.0, run_seed=rs, nt=NT)
        offs = np.asarray(o, float); pf = np.asarray(f, float) / np.asarray(n, float)
        pk = float(np.nanmax(pf)); above = offs[pf >= 0.5 * pk]
        tbws.append(float(np.nanmax(above) - np.nanmin(above)) if above.size >= 2 else float("nan"))
        pks.append(pk)
    tbw_med = float(np.nanmedian(tbws))

    fp_after = wfp(net)
    bit_identical = bool(fp_before == fp_after)
    fwa = fw("AFTER")
    fw_ok = bool(fwa == fwb)
    assert bit_identical, f"WEIGHTS MUTATED {fp_before}!={fp_after}"
    assert fw_ok, "FIREWALL DRIFT"

    print(f"  [TBW] {[f'{x:.0f}' for x in tbws]} med={tbw_med:.0f} peakP {[f'{p:.2f}' for p in pks]} "
          f"| weight_bit_identical={bit_identical} firewall_ok={fw_ok}", flush=True)
    print(f"### {A.tag}: aM={eff['aM']:.4f} dM={eff['dM']:.0f} pv_gaba={eff['pv_gaba_scale']:.2f} "
          f"tau_gaba={eff['tau_gaba']:.0f} gNMDA={eff['gNMDA']:.3f} -> TBWmed={tbw_med:.0f} TBW={[round(x) for x in tbws]} "
          f"bit_id={bit_identical} fw_ok={fw_ok} ###", flush=True)

    out = dict(tag=A.tag, ckpt=A.ckpt, md5=md5, meas_seed=MEAS_SEED, effective=eff,
               tbw=tbws, tbw_med=tbw_med, tbw_peak=pks,
               weight_fp_before=fp_before, weight_fp_after=fp_after, weight_bit_identical=bit_identical,
               md5_TBW=fwa[0], md5_SBW=fwa[1], firewall_ok=fw_ok)
    if A.out:
        os.makedirs(os.path.dirname(A.out), exist_ok=True)
        json.dump(out, open(A.out, "w"), indent=1)


if __name__ == "__main__":
    main()
