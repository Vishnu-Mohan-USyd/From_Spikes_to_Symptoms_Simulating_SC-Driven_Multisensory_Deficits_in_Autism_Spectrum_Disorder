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

# Paths self-locate from this file's location so the tree runs from any clone path.
# (Flat layout: this file, the Training builds, the frozen readouts, checkpoint/ and out/ all share one dir.)
# Override the project root with $FSTS_PERILOG if the tree is rearranged.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))                              # repo root (flat layout)
ROOT = _THIS_DIR  # flat: project root = this directory
_FSTS5 = os.path.dirname(ROOT)                                                      # .../fsts_5
CODE = ROOT                                                  # canonical Training.py 466c9a76
DELAYFIX = os.path.join(ROOT, "Training_delayfix_d52.py")  # delay52 MEASUREMENT build (matches delay52 ckpt forward)
TBW_DIR = ROOT              # TBW_test 80d33465
SBW_DIR = ROOT  # SBW_test 73b7d136


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
        net.gNMDA = float(os.environ.get("GNMDA", 1.30)); net.tau_nmda = float(os.environ.get("TAU_NMDA", "80.0")); net.nmda_alpha = 0.1   # #66 gNMDA + NR2A TAU_NMDA env (load_ckpt restore wins post-load; defaults 1.30/80.0 == byte-identical)
        net.tau_nmda_v = float(os.environ.get("TAU_NMDA_V", net.tau_nmda))   # latency #153: V-specific FF-NMDA decay (measure build_net; default == net.tau_nmda => byte-identical; load_ckpt re-syncs off the ckpt-restored tau_nmda)
        net.Erev_nmda = 20.0; net.tau_nmdaVolt = 100.0; net.v_nmda_rest = -65.0
        net.nmda_vrest_offset = 7.0; net.mg_vhalf = float(os.environ.get("MG_VHALF", -35.0)); net.tau_nmda_inh = 21.6
        net.dend_coupling_alpha = float(os.environ.get("DEND_COUPLING_ALPHA", 0.1))   # P5 NMDA Mg-gate knob (env-config; default == original 0.1)
        net.mg_k = float(os.environ.get("MG_K", 0.062))                               # C1 NMDA Mg-gate slope (env-config; default == original 0.062)
        net.mg_vhalf_inh = float(os.environ.get("MG_VHALF_INH", net.mg_vhalf))         # inh-gate DECOUPLE (env-config; default == exc mg_vhalf => byte-identical)
    net.u_a.fill_(net.u_stp_a); net.u_v.fill_(net.u_stp_v); net.tau_rec = float(os.environ.get("TAU_REC", 400.0))  # task#104: dead 0.7 fill -> env-exposed operative U (reset_state re-applies); tau_rec env-exposed (defaults 0.2/400 byte-identical)
    net.input_scaling = 400; net.g_GABA = float(os.environ.get("G_GABA", "10"))   # #94/#92 inhibition lever (env-config; default == original 10 => byte-identical; load_ckpt restore wins post-load)
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
    # task#P6: tau_gaba is a plain attr (NOT in model_state). make_checkpoint saves the TRAINED value
    # into mutable_hparams, so the CKPT is the source of truth -> measurement matches training no matter
    # the env. Legacy/raw-state_dict ckpts (no mutable_hparams) keep the build default with a WARN.
    mh = ck.get("mutable_hparams", {})
    if "tau_gaba" in mh:
        net.tau_gaba = float(mh["tau_gaba"])
    else:
        print(f"[load_ckpt] WARN: ckpt has no mutable_hparams['tau_gaba'] -> tau_gaba falls back to "
              f"build default {float(net.tau_gaba)} ms", flush=True)
    # if TAU_GABA env is set it MUST agree with the ckpt (catch a deliberate/typo mismatch); if unset,
    # the ckpt value silently wins (env no longer required to measure correctly).
    _env_tau = os.environ.get("TAU_GABA")
    if _env_tau is not None:
        assert abs(float(net.tau_gaba) - float(_env_tau)) < 1e-9, \
            f"FATAL: ckpt tau_gaba={float(net.tau_gaba)} != TAU_GABA env {_env_tau}"
    # task#68: gNMDA is a plain attr (NOT in model_state), set by the build recipe to the build default.
    # make_checkpoint saves the TRAINED gain into mutable_hparams, so the CKPT is the source of truth ->
    # a 0.70-trained cheap-prove is graded at 0.70 regardless of the build default/env. Fixes the
    # grading-integrity bug where load_ckpt kept the 1.30 build default (cooled weights judged under hot
    # drive = corrupt TBW/SBW/IE/E-I). Mirrors the tau_gaba round-trip above.
    if "gNMDA" in mh:
        net.gNMDA = float(mh["gNMDA"])
    else:
        print(f"[load_ckpt] WARN: ckpt has no mutable_hparams['gNMDA'] -> gNMDA falls back to "
              f"build default {float(net.gNMDA)}", flush=True)
    _env_gnmda = os.environ.get("GNMDA")
    if _env_gnmda is not None:
        assert abs(float(net.gNMDA) - float(_env_gnmda)) < 1e-9, \
            f"FATAL: ckpt gNMDA={float(net.gNMDA)} != GNMDA env {_env_gnmda}"
    # task#73-combo (MEASURE-side, mirrors gNMDA #68): tau_nmda is a plain attr (build hardcodes 80.0), but
    # make_checkpoint ALREADY saves the TRAINED tau_nmda into mutable_hparams (graphdf L4725) -> the CKPT is the
    # source of truth so the combined retrain's NR2A tau (e.g. 50ms) is graded at its trained value, not 80ms.
    # Existing ckpts trained at 80 restore to 80 => no regression. TAU_NMDA env (if set) must agree.
    if "tau_nmda" in mh:
        net.tau_nmda = float(mh["tau_nmda"])
    else:
        print(f"[load_ckpt] WARN: ckpt has no mutable_hparams['tau_nmda'] -> tau_nmda falls back to "
              f"build default {float(net.tau_nmda)} ms", flush=True)
    _env_taunmda = os.environ.get("TAU_NMDA")
    if _env_taunmda is not None:
        assert abs(float(net.tau_nmda) - float(_env_taunmda)) < 1e-9, \
            f"FATAL: ckpt tau_nmda={float(net.tau_nmda)} != TAU_NMDA env {_env_taunmda}"
    # latency #153: re-sync V-specific FF-NMDA decay off the FINAL (ckpt-restored) tau_nmda. This is the
    # OPERATIVE finalization for the measure/probe path (build_net then load_ckpt). Default (TAU_NMDA_V unset)
    # => tau_nmda_v == net.tau_nmda => bit-identical default branch; TAU_NMDA_V env > tau_nmda => split path
    # (the debugger's #152 read-only sweep knob). gNMDA/Mg-block/A-pathway/recurrent+inhib NMDA all untouched.
    net.tau_nmda_v = float(os.environ.get("TAU_NMDA_V", net.tau_nmda))
    # task#94 (MEASURE-side, mirrors gNMDA #68): g_GABA is the #92 inhibition lever, saved to mutable_hparams
    # (graphdf L4711) but built to the default 10. Restore the TRAINED value so a g_GABA-fix ckpt is graded at
    # its trained inhibition; G_GABA env (if set) must agree. Existing ckpts (g_GABA=10) restore to 10 => no regression.
    if "g_GABA" in mh:
        net.g_GABA = float(mh["g_GABA"])
    else:
        print(f"[load_ckpt] WARN: ckpt has no mutable_hparams['g_GABA'] -> g_GABA falls back to "
              f"build default {float(net.g_GABA)}", flush=True)
    _env_ggaba = os.environ.get("G_GABA")
    if _env_ggaba is not None:
        assert abs(float(net.g_GABA) - float(_env_ggaba)) < 1e-9, \
            f"FATAL: ckpt g_GABA={float(net.g_GABA)} != G_GABA env {_env_ggaba}"
    # task#94 (MEASURE-side, mirrors gNMDA #68): mg_k IS saved to mutable_hparams + env-exposed in both builds,
    # but was NOT restored -> a mg_k-SWEEP ckpt would be graded at the build/env mg_k (e.g. a grader's stale 0.15),
    # not its trained value. CKPT is the source of truth; MG_K env (if set) must agree (catches a stale hardcode).
    if "mg_k" in mh:
        net.mg_k = float(mh["mg_k"])
    else:
        print(f"[load_ckpt] WARN: ckpt has no mutable_hparams['mg_k'] -> mg_k falls back to "
              f"build default {float(net.mg_k)}", flush=True)
    _env_mgk = os.environ.get("MG_K")
    if _env_mgk is not None:
        assert abs(float(net.mg_k) - float(_env_mgk)) < 1e-9, \
            f"FATAL: ckpt mg_k={float(net.mg_k)} != MG_K env {_env_mgk}"
    # task#139 (MEASURE-side, mirrors mg_k): conduction_delay_v2msi is the V-advance LATENCY lever (substeps),
    # saved to mutable_hparams + env-exposed (CONDUCTION_DELAY_V2MSI) in both builds. Restore the TRAINED value
    # (CKPT = source of truth) and RECOMPUTE the derived _ms / _inh / _inh_ms so the eager forward reads the
    # trained V arrival delay even when the grader env is absent; CONDUCTION_DELAY_V2MSI env (if set) must agree.
    if "conduction_delay_v2msi" in mh:
        net.conduction_delay_v2msi = int(mh["conduction_delay_v2msi"])
        net.conduction_delay_v2msi_inh = net.conduction_delay_v2msi + 20
        net.conduction_delay_v2msi_ms = net.conduction_delay_v2msi * net.dt
        net.conduction_delay_v2msi_inh_ms = net.conduction_delay_v2msi_inh * net.dt
    else:
        print(f"[load_ckpt] WARN: ckpt has no mutable_hparams['conduction_delay_v2msi'] -> v2msi falls back to "
              f"build default {int(net.conduction_delay_v2msi)} substeps", flush=True)
    _env_v2msi = os.environ.get("CONDUCTION_DELAY_V2MSI")
    if _env_v2msi is not None:
        assert int(net.conduction_delay_v2msi) == int(_env_v2msi), \
            f"FATAL: ckpt conduction_delay_v2msi={int(net.conduction_delay_v2msi)} != CONDUCTION_DELAY_V2MSI env {_env_v2msi}"
    # task#102 (B, MEASURE-side, mirrors g_GABA/mg_k): msiInh_input_gain is the PERSISTENT FFI-recruitment forward-
    # multiplier, saved to mutable_hparams + env-exposed (MSIINH_INPUT_GAIN) in both builds, but built to default 1.0.
    # Restore the TRAINED value so an FFI-recruitment ckpt is graded at its trained recruitment; MSIINH_INPUT_GAIN env
    # (if set) must agree (catches a stale grade). Old ckpts (no mh entry) restore to the build default 1.0 => byte-id.
    if "msiInh_input_gain" in mh:
        net.msiInh_input_gain = float(mh["msiInh_input_gain"])
    else:
        print(f"[load_ckpt] WARN: ckpt has no mutable_hparams['msiInh_input_gain'] -> falls back to "
              f"build default {float(net.msiInh_input_gain)}", flush=True)
    _env_ffig = os.environ.get("MSIINH_INPUT_GAIN")
    if _env_ffig is not None:
        assert abs(float(net.msiInh_input_gain) - float(_env_ffig)) < 1e-9, \
            f"FATAL: ckpt msiInh_input_gain={float(net.msiInh_input_gain)} != MSIINH_INPUT_GAIN env {_env_ffig}"
    # task#104 (MEASURE-side, mirrors g_GABA/mg_k/FFI(B)): the STD/adaptation decouple knobs are saved to
    # mutable_hparams + env-exposed in both builds (TM-STD: tau_rec / u_stp_a / u_stp_v / nmda_std_scale;
    # spike-adaptation: bM / dM), all built to their CURRENT defaults (byte-identical). Restore the TRAINED value so a
    # decouple-retrain ckpt is graded at its trained STD/adaptation; the matching env (if set) must AGREE (catches a
    # stale grade). Old ckpts lacking an entry restore to the build default => byte-identical. (u_stp_a/v are scalars;
    # the operative u_a/u_v tensors are re-filled from them by the reset_state below, so restoring the scalar suffices.)
    for _mhk, _attr, _envn in [("tau_rec", "tau_rec", "TAU_REC"),
                               ("u_stp_a", "u_stp_a", "U_STP_A"),
                               ("u_stp_v", "u_stp_v", "U_STP_V"),
                               ("nmda_std_scale", "nmda_std_scale", "NMDA_STD_SCALE"),
                               ("bM", "bM", "ADAPT_BM"),
                               ("dM", "dM", "ADAPT_DM"),
                               ("gAMPA_LP", "gAMPA_LP", "GAMPA_LP"),    # task#116 orthogonal fast-exc rate lever
                               ("k_dvdt", "k_dvdt", "K_DVDT"),          # task#130 Form-A dVm/dt-adaptive threshold (Azouz&Gray 2000) latency fix
                               ("tau_dvdt", "tau_dvdt", "TAU_DVDT"),
                               ("v_thresh_floor", "v_thresh_floor", "V_THRESH_FLOOR"),
                               ("dvdt_cap", "dvdt_cap", "DVDT_CAP")]:
        if _mhk in mh:
            setattr(net, _attr, float(mh[_mhk]))
        else:
            print(f"[load_ckpt] WARN: ckpt has no mutable_hparams['{_mhk}'] -> {_attr} falls back to "
                  f"build default {float(getattr(net, _attr))}", flush=True)
        _envv = os.environ.get(_envn)
        if _envv is not None:
            assert abs(float(getattr(net, _attr)) - float(_envv)) < 1e-9, \
                f"FATAL: ckpt {_attr}={float(getattr(net, _attr))} != {_envn} env {_envv}"
    # task#130 Form-A: re-derive the toggle from the RESTORED k_dvdt. The loop above can set k_dvdt from mh AFTER
    # __init__ already computed _formA_on at construction; without this, a Form-A ckpt loaded with K_DVDT env UNSET
    # would measure with the adaptive rule silently OFF (wrong grade). k_dvdt==0 => OFF (byte-identical). Ckpt = truth.
    net._formA_on = (float(net.k_dvdt) != 0.0)
    # task#73-combo (MEASURE-side): g_rec is NOT in mutable_hparams (it is a scheduled training operating point),
    # so for a combined retrain trained at a TRIMMED g_rec (e.g. 0.03) the measurement must be told via the G_REC
    # env; unset -> the canonical per-epoch schedule (0.1) wins, byte-identical for every existing grade. Early
    # warmup (ep<=25) keeps g_rec=0.0 regardless. [g_rec stays scheduled train-side; this only lets MEASURE honor
    # the trained trim. Contract: g_rec=0.03 is the constant post-warmup operating point — confirm w/ debugger.]
    _env_grec = os.environ.get("G_REC")
    if (epoch is None or int(epoch) > 25):
        g_rec = float(_env_grec) if _env_grec is not None else 0.1
    else:
        g_rec = 0.0
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
        # task #55: shunt-aware inhibition denominator. Under GABA_SHUNT_SURR the lateral surround leaves
        # I_M_gaba for the divisive I_surr_shunt term, so add it back: <I_M_gaba - I_surr_shunt> (I_surr_shunt<=0).
        # I_surr_shunt==0 when shunt OFF (and absent on the canonical no-shunt build) => identical to the
        # historical stock denominator (backward-compatible; proven shunt-aware===stock to 0 on the gate#1 OFF ckpt).
        _iss = getattr(net, "_last_I_surr_shunt", None)
        if _iss is None:
            iMg = net.I_M_gaba.detach().clamp(min=0)
        else:
            iMg = (net.I_M_gaba.detach() - _iss.detach()).clamp(min=0)
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
          f"tau_gaba={float(net.tau_gaba)} "
          f"plast={net.plasticity_enabled} comment={comment!r}", flush=True)
    res = dict(tag=args.tag, ckpt=args.ckpt, seed=args.seed, epoch=epoch,
               build=BUILD, tau_nmda_inh=float(net.tau_nmda_inh), g_rec=float(net.g_rec),
               tau_gaba=float(net.tau_gaba),
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
