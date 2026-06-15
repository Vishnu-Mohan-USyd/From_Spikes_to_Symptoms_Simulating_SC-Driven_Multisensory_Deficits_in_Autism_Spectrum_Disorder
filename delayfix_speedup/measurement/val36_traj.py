#!/usr/bin/env python3
"""TASK #45 validator harness — TBW / SBW / E-I on ONE delay-fix retrain checkpoint.

READ-ONLY measurement (validator). No production code or apparatus is edited; the
delay-fix net build + the route-c apparatus modules are IMPORTED (is_temporally_fused
UNTOUCHED). One checkpoint per process so epochs/seeds run in parallel.

WHY the delay-fix build (Training_delayfix.py md5 5e7d6d20), not canonical Training.py:
  These checkpoints were TRAINED with the delayed-recurrence forward (EDIT#1: the g_rec
  injection reads buffer_msi_rec — MSI spikes delayed by 1 external step / 10 ms — instead
  of the instantaneous new_sM). Measuring delay-fixed weights with the canonical
  (instantaneous-recurrence) forward would be a confound, so we build the SAME forward the
  weights were trained under. preflight_ckpt.py proved the delayfix state_dict key-set ==
  canonical (buffer_msi_rec / _prev_sMSI_rec are plain attrs, not serialized), so the
  apparatus (which `from Training import *`) binds cleanly to the delayfix class.

CKPT schema (make_checkpoint): {model_state, constructor_hparams, mutable_hparams, epoch,...}.
  tau_nmda_inh is in NEITHER hparam dict (it is a plain attr lost on save), so a rebuild
  defaults it to the constructor 45.0 — we FORCE 21.6 on every load (lead mandate). We rebuild
  via the EXACT training recipe build_net(seed) (so single_modality_prob / sigma_teacher and
  every fixed hyperparam match what trained the weights) then strict-load model_state.

  g_rec is the per-epoch training operating point: 0.0 for ep<=25 (pre-ungate), 0.1 for ep>25
  (retrain_delayfix loop: `net.g_rec = 0.1 if ep > 25 else 0.0`). We mirror it from the ckpt's
  epoch so the measured dynamics match the trained operating point at that epoch.

Reference bars (reported, not all gated — lead: trajectory SHAPE matters, E/I scale under review):
  TBW robust-FWHM 396-419 ms / SDrob 168-178 ms (human 414/176); SBW half-width band
  [24.5, 40.9] deg (route-c ref ~31.1); E/I PROVISIONAL (panel 13-24 vs VAL35 ~6.3/0.58 vs
  biology ~1 — debugger #46/#47 reconciling; report value + trajectory shape, do not gate).
"""
import os, sys, json, argparse, importlib.util, hashlib, time
import numpy as np
import torch
import random

ROOT = "/home/vishnu/coding_proj/fsts_5/fsts_perilog_20260607"
CODE = os.path.join(ROOT, "code")                                                  # canonical Training.py 466c9a76
DELAYFIX = os.path.join(ROOT, "delayfix_retrain_20260614", "Training_delayfix.py")  # md5 5e7d6d20
TBW_DIR = "/home/vishnu/coding_proj/fsts_5/repo/route_c_tbw_delivery/code"          # TBW_test 80d33465
SBW_DIR = "/home/vishnu/coding_proj/fsts_5/repo/route_c_tbw_delivery/sbw/apparatus"   # SBW_test 73b7d136


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m          # register so each module's `from Training import *` binds here
    spec.loader.exec_module(m)
    return m


# ── choose the net definition (default: delay-fix build, the faithful forward) ──
BUILD = os.environ.get("VAL36_BUILD", "delayfix")
NETSRC = DELAYFIX if BUILD == "delayfix" else os.path.join(CODE, "Training.py")
# Register the SAME module object under BOTH 'Training' (for the apparatus `from Training import *`)
# and 'Training_delayfix' — one object, so the net class the apparatus sees == the one we build.
T = load_mod("Training", NETSRC)
sys.modules["Training_delayfix"] = T
# apparatus AFTER Training is registered
TBW = load_mod("TBW_test", os.path.join(TBW_DIR, "TBW_test.py"))
SBW = load_mod("SBW_test", os.path.join(SBW_DIR, "SBW_test.py"))


# ── build_net = retrain_delayfix.build_net, VERBATIM (the recipe that trained the ckpts) ──
def build_net(seed, batch_size=1000):
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    net = T.MultiBatchAudVisMSINetworkTime(
        n_neurons=180, batch_size=batch_size,
        lr_unimodal=2e-2, lr_msi=2e-2, lr_readout=8e-4,
        sigma_in=10.0, sigma_teacher=2.0, noise_std=0.02, single_modality_prob=0.5,
        v_thresh=0.3, dt=0.1, tau_m=20.0, n_substeps=100, loc_jitter_std=0,
        space_size=180, conduction_delay_a2msi=250, conduction_delay_v2msi=400)
    net.set_inhib_plasticity(True)
    T.assign_unimodal_preferred_locations(net)
    net.b_uniA.data.fill_(0.0); net.b_uniV.data.fill_(0.0)
    T._init(net.W_a2msi_AMPA, 0.004); T._init(net.W_v2msi_AMPA, 0.004)
    T._init(net.W_a2msi_NMDA, 0.004); T._init(net.W_v2msi_NMDA, 0.004)
    with torch.no_grad():
        net.gNMDA = 1.30; net.tau_nmda = 80.0; net.nmda_alpha = 0.1
        net.Erev_nmda = 20.0; net.tau_nmdaVolt = 100.0; net.v_nmda_rest = -65.0
        net.nmda_vrest_offset = 7.0; net.mg_vhalf = -35.0; net.tau_nmda_inh = 21.6
    net.u_a.fill_(0.7); net.u_v.fill_(0.7); net.tau_rec = 400.0
    net.input_scaling = 400; net.g_GABA = 10
    return net


def load_ckpt(ckpt_path, seed, B, device):
    ck = torch.load(ckpt_path, map_location=device)
    if "model_state" not in ck:           # tolerate a raw state_dict too
        ck = {"model_state": ck, "epoch": None}
    epoch = ck.get("epoch", None)
    net = build_net(seed, batch_size=B)
    res = net.load_state_dict(ck["model_state"], strict=True)
    # FORCE the lead-mandated knobs (ckpt carries neither tau_nmda_inh nor g_rec):
    net.tau_nmda_inh = 21.6
    g_rec = 0.1 if (epoch is None or int(epoch) > 25) else 0.0   # per-epoch training operating point
    net.g_rec = g_rec
    net.to(device); net.eval(); net.plasticity_enabled = False
    assert abs(float(net.tau_nmda_inh) - 21.6) < 1e-9
    return net, res, epoch, g_rec, ck.get("comment", "")


# ── TBW: route-c temporal-fusion curve + robust fit + BOX/PLATEAU characterization ──
def _edges(offs_ms, pf, thr):
    """[lo,hi] ms where pf >= thr (None if never crosses)."""
    idx = np.where(pf >= thr)[0]
    if len(idx) == 0:
        return None, None, 0.0
    lo, hi = float(offs_ms[idx.min()]), float(offs_ms[idx.max()])
    return lo, hi, hi - lo


def measure_tbw(net, n_trials=50):
    offsets = list(range(-30, 31, 2))                  # -300..+300 ms in 20 ms steps (31 pts)
    out = TBW.compute_tbw_temporal_fusion_persep(
        net, offsets, n_trials=n_trials, T=60, D=5, stim_in=1.0,
        sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
        min_peak_separation=3, min_total=10.0)
    pf = np.asarray(out[0], float)
    offs_ms = np.array([o * 10 for o in offsets], float)
    fit = TBW.fit_psychometric_curve(offs_ms, pf, robust_fit=True)
    i0 = offsets.index(0)
    lo50, hi50, w50 = _edges(offs_ms, pf, 0.5)         # box edges (robust width descriptor)
    lo95, hi95, w95 = _edges(offs_ms, pf, 0.95)        # plateau (flat-top) extent
    peak = float(pf.max())
    # shape: a saturated wide flat top => box/plateau; a smooth sub-saturating hump => graded
    flat_frac = (w95 / w50) if (w50 and w50 > 0) else 0.0
    if peak >= 0.95 and flat_frac >= 0.5:
        shape = "box/plateau"
    elif peak >= 0.5:
        shape = "graded"
    else:
        shape = "degenerate/low"
    return dict(offsets_ms=offs_ms.tolist(), pfusion=pf.tolist(),
                fwhm_ms=float(fit["fwhm"]), sdrob_ms=float(fit["sigma"]),
                mu_ms=float(fit["mu"]), r_squared=float(fit["r_squared"]),
                peak=peak, p_at_0=float(pf[i0]),
                box_lo50_ms=lo50, box_hi50_ms=hi50, box_width50_ms=w50,
                plateau_lo95_ms=lo95, plateau_hi95_ms=hi95, plateau_width95_ms=w95,
                shape=shape)


# ── SBW: route-c spatial-fusion curve + pedestal half-width ──
def measure_sbw(net, n_trials=50):
    seps = list(range(-80, 85, 5))                     # 33 pts (#39 grid)
    pf = np.asarray(SBW.compute_sbw_fused_persep(
        net, separations_deg=seps, n_trials=n_trials, intensity=0.5, duration=20), float)
    try:
        _xs, _ys, popt = SBW.fit_pedestal_curve(
            {"separations_deg": np.array(seps, float), "mean_prob": pf})
        hw = float(abs(popt[2]))
    except Exception as e:
        hw = float("nan")
        print("[SBW] fit fail: %r" % e, flush=True)
    in_band = bool(24.5 <= hw <= 40.9) if hw == hw else False
    return dict(separations_deg=seps, pfusion=pf.tolist(), halfwidth_deg=hw, in_band_24p5_40p9=in_band)


# ── E/I: route-c <I_M>/<I_M_gaba> (replicated ei_probe_route_c; no_grad not inference_mode) ──
@torch.no_grad()
def ei_probe_route_c(net, *, offset_steps, centre_deg, sigma_in, pulse_frames, n_frames, intensity):
    B, N = 1, net.n
    net.reset_state(batch_size=B)
    net.start_ei_recording()
    xA = torch.zeros(n_frames, N, device=net.device)
    xV = torch.zeros_like(xA)
    idx_c = int(round(centre_deg * (N - 1) / (net.space_size - 1)))
    xs = torch.arange(N, dtype=torch.float32, device=net.device)
    gauss = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * intensity
    aud_start = 0 if offset_steps >= 0 else abs(offset_steps)
    vis_start = 0 if offset_steps <= 0 else offset_steps
    xA[aud_start:aud_start + pulse_frames] = gauss
    xV[vis_start:vis_start + pulse_frames] = gauss
    I_M_tr, I_Mg_tr, v_tr = [], [], []
    for t in range(n_frames):
        net.update_all_layers_batch(xA[t].unsqueeze(0), xV[t].unsqueeze(0), valid_mask=None)
        iM = net.I_M.detach().clamp(min=0)
        iMg = net.I_M_gaba.detach().clamp(min=0)
        I_M_tr.append(iM.mean().item())
        I_Mg_tr.append(iMg.mean().item())
        v_tr.append(net.v_msi.detach().mean().item())
    rec = net.stop_ei_recording()
    return np.asarray(I_M_tr), np.asarray(I_Mg_tr), np.asarray(v_tr), rec


def measure_ei(net):
    offsets = [-30, -20, -10, 0, 10, 20, 30]
    sigma_in = float(getattr(net, "sigma_in", 10.0))
    exc, inh, vrep, comp_ie, comp_ii = [], [], [], [], []
    for off in offsets:
        I_M_tr, I_Mg_tr, v_tr, rec = ei_probe_route_c(
            net, offset_steps=off, centre_deg=90.0, sigma_in=sigma_in,
            pulse_frames=5, n_frames=20, intensity=1.0)
        exc.append(float(I_M_tr.mean())); inh.append(float(I_Mg_tr.mean()))
        onset = abs(off)
        vseg = v_tr[onset:] if onset < len(v_tr) else v_tr
        vrep.append(float(np.mean(vseg)))
        ie = float(np.mean(rec["I_E_mean"])) if len(rec.get("I_E_mean", [])) else float("nan")
        ii = float(np.mean(rec["I_I_mean"])) if len(rec.get("I_I_mean", [])) else float("nan")
        comp_ie.append(ie); comp_ii.append(ii)
    exc = np.asarray(exc); inh = np.asarray(inh)
    ratios = exc / (inh + 1e-12)
    j0 = offsets.index(0)
    comp_ie = np.asarray(comp_ie); comp_ii = np.asarray(comp_ii)
    comp_ei = float(np.nanmean(comp_ie / (comp_ii + 1e-12)))
    return dict(offsets_frames=offsets, per_offset_ei=ratios.tolist(),
                # PRIMARY (debugger #46 ruling): accumulated MEMBRANE ratio <I_M>/<I_M_gaba>,
                # the current the Izhikevich membrane integrates (dVM += I_M - I_M_gaba, same
                # units/site). Target ~= 1. Canonical baseline mildly inhibition-dominated ~0.46-0.58.
                ei_ratio_sync=float(ratios[j0]), ei_ratio_offsetmean=float(ratios.mean()),
                # DIAGNOSTIC ONLY — source-side estimator, driving-force(+tau)-confounded, NOT
                # membrane balance (proven artifact: tau_gaba 50->2.5 flips membrane ratio 0.46->3.50).
                comp_ei_offsetmean_DIAG_sourceSide_confounded=comp_ei,
                exc_offsetmean=float(exc.mean()), inh_offsetmean=float(inh.mean()),
                v_rep_offsetmean=float(np.mean(vrep)),
                ei_primary_metric="membrane <I_M>/<I_M_gaba> (ei_ratio_sync/offsetmean), target~1",
                note="PRIMARY=membrane ratio vs ~1 (debugger #46). comp_ei=source-side confounded diagnostic only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epoch", type=int, default=None, help="override; else read from ckpt['epoch']")
    ap.add_argument("--metrics", default="tbw,sbw,ei")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"[{args.tag}] device={device} BUILD={BUILD} netsrc.md5={md5(NETSRC)} "
          f"TBW.md5={md5(os.path.join(TBW_DIR,'TBW_test.py'))} "
          f"SBW.md5={md5(os.path.join(SBW_DIR,'SBW_test.py'))}", flush=True)
    net, ldres, ck_epoch, g_rec, comment = load_ckpt(args.ckpt, args.seed, 256, device)
    epoch = args.epoch if args.epoch is not None else ck_epoch
    if args.epoch is not None:            # honor override for g_rec too
        net.g_rec = 0.1 if int(args.epoch) > 25 else 0.0
        g_rec = float(net.g_rec)
    print(f"[{args.tag}] loaded ckpt={os.path.basename(args.ckpt)} epoch={epoch} "
          f"missing={list(ldres.missing_keys)} unexpected={list(ldres.unexpected_keys)} "
          f"tau_nmda_inh={float(net.tau_nmda_inh)} g_rec={float(net.g_rec)} "
          f"plast={net.plasticity_enabled} comment={comment!r}", flush=True)
    res = dict(tag=args.tag, ckpt=args.ckpt, seed=args.seed, epoch=epoch,
               build=BUILD, tau_nmda_inh=float(net.tau_nmda_inh), g_rec=float(net.g_rec),
               plasticity_enabled=bool(net.plasticity_enabled),
               missing_keys=list(ldres.missing_keys), unexpected_keys=list(ldres.unexpected_keys),
               md5_netsrc=md5(NETSRC),
               md5_TBW_test=md5(os.path.join(TBW_DIR, "TBW_test.py")),
               md5_SBW_test=md5(os.path.join(SBW_DIR, "SBW_test.py")))
    want = [w.strip() for w in args.metrics.split(",") if w.strip()]
    with torch.no_grad():           # one uniform no_grad: no autograd graph, no inference-tensor tagging
        if "ei" in want:
            res["ei"] = measure_ei(net); print(f"[{args.tag}] EI done {time.time()-t0:.1f}s "
                                                f"E/I(sync)={res['ei']['ei_ratio_sync']:.3f} "
                                                f"E/I(offmean)={res['ei']['ei_ratio_offsetmean']:.3f}", flush=True)
        if "tbw" in want:
            res["tbw"] = measure_tbw(net); t = res["tbw"]
            print(f"[{args.tag}] TBW done {time.time()-t0:.1f}s shape={t['shape']} "
                  f"box50=[{t['box_lo50_ms']},{t['box_hi50_ms']}]={t['box_width50_ms']}ms "
                  f"plateau95={t['plateau_width95_ms']}ms FWHMfit={t['fwhm_ms']:.0f} "
                  f"peak={t['peak']:.3f} P@0={t['p_at_0']:.3f}", flush=True)
        if "sbw" in want:
            res["sbw"] = measure_sbw(net); s = res["sbw"]
            print(f"[{args.tag}] SBW done {time.time()-t0:.1f}s hw={s['halfwidth_deg']:.2f}deg "
                  f"in_band={s['in_band_24p5_40p9']} Pf={min(s['pfusion']):.2f}..{max(s['pfusion']):.2f}", flush=True)
    json.dump(res, open(args.out_json, "w"), indent=1)
    print(f"[{args.tag}] DONE {time.time()-t0:.1f}s -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
