#!/usr/bin/env python3
"""task #105 — per-seed 5-seed VALIDATION retrain with the correlated common-mode trial-timing-noise (ΔL) fix.

Scale-up of the #104 develop-check (Rule-3 gate PASSED at ep50: box->graded bell, max_step 0.244, K1 false,
SBW in-band, no jitter-induced EI/MSI regression). ONE seed, ep0->79, run in ITS OWN PROCESS (one L6-graphed
net per process -> dodges the #55/#68 two-graph segfault). The ONLY change vs the certified delay52/tau10/bs250
reference lineage (retrain_delay52.py) is SIGMA_DL_FRAMES=3 (per-trial COMMON-MODE A-V latency ΔL~N(0,3
frames=30ms), baked INTRINSICALLY into the TRAINING generator). Byte-protocol-identical otherwise:
  seed-once-then-loop, g_rec=0.1 if ep>25 else 0.0, n_seq=1000, bs=250 (4x250 no partial),
  W_MSI_exc.fill_diagonal_(0) each epoch, CERTIFIED L6 levers (L1=0, L2=1, L6=1, L4 substrate).

Checkpoints ep5 / ep30 / ep79 in RUN_DIR (separate from the #104 develop-check ckpts). The independent
ep79 TBW/SBW/EI scientific measurement is the VALIDATOR's job (frozen readout md5 80d33465) — this driver
does ONLY training + the per-epoch inline vitals/kill battery + ckpts. Exits 1 on any KILL (launcher counts).

INLINE KILL BATTERY (graph-available; verbatim from the develop-check, Lead-signed-off):
  * NaN/Inf in any monitored weight                              -> KILL
  * [#106 REMOVED — mis-calibrated] activity collapse _g_out_sMSI snapshot < MSI_SNAP_COLLAPSE: the proxy
    floor sits inside the normal late-epoch low-rate band (healthy no-jitter ref runs there) -> NOT a kill;
    snapshot still logged/banked. Canonical-MSI sanity is the ep30 post-hoc (eval-path Hz vs no-jitter base).
  * weight-norm blow-up > NORM_BLOWUP x init (W_msiInh2Exc_GABA exempt up to GABA_BLOWUP: ramps by design)
  * ep5 iSTDP-GABA over-saturation: clampfrac >= EP5_CLAMPFRAC_KILL (knob=0 baseline 0.0000)
ep30 canonical MSI / n_active vs matched no-jitter baseline + ep79 TBW/SBW/EI are POST-HOC on the ckpts
(canonical _dbg rate .item()-syncs -> CUDA-graph capture skips it -> NaN during graphed training).

Usage:  SIGMA_DL_FRAMES=3 TAU_GABA=10 CUDA_VISIBLE_DEVICES=<g> python retrain5_jitter.py SEED
"""
import os, sys, time, json, random, importlib
import numpy as np
import torch
import faulthandler; faulthandler.enable()
import panelgraph as PG

GDF = importlib.import_module("Training_graphdf_d52")   # delay-corrected fast graphed build (reference lineage)

SEED      = int(sys.argv[1]) if len(sys.argv) > 1 else 42
BS        = 250
N_SEQ     = 1000
N_EPOCHS  = int(os.environ.get('N_EPOCHS_OVERRIDE', '80'))   # ep0..ep79 (override only for the cheap pre-launch smoke)
TAU_GABA  = float(os.environ.get('TAU_GABA', '10.0'))       # reference delay52 tau10 lineage
SIGMA_DL  = float(os.environ.get('SIGMA_DL_FRAMES', '3.0'))  # THE fix: 3 frames = 30 ms common-mode
TAU_TAG   = ('%g' % TAU_GABA).replace('.', 'p')
SIG_TAG   = ('%g' % SIGMA_DL).replace('.', 'p')
THIS      = os.path.dirname(os.path.abspath(__file__))
RUN_DIR   = os.environ.get("RUN_DIR_OVERRIDE", os.path.join(THIS, "checkpoint"))     # dedicated 5-seed artifact dir (RUN_DIR_OVERRIDE -> scratch for P5 cheap-prove, no clobber of dL3 ckpts)
os.makedirs(RUN_DIR, exist_ok=True)
def _ck(ep): return os.path.join(RUN_DIR, f"ckpt_ep{ep}_seed{SEED}_bs250_delay52_tau{TAU_TAG}_dL{SIG_TAG}.pt")
CKPT_EP   = {5: _ck(5), 30: _ck(30), 50: _ck(50), 79: _ck(79)}   # ep50 added (Lead: cross-check vs develop-check)
LOG_PATH  = os.path.join(RUN_DIR, f"retrain5_seed{SEED}_tau{TAU_TAG}_dL{SIG_TAG}.log")
JSON_PATH = os.path.join(RUN_DIR, f"retrain5_seed{SEED}_tau{TAU_TAG}_dL{SIG_TAG}.json")
GATE_EPS  = set(range(0, N_EPOCHS, 5)) | {N_EPOCHS - 1}     # 0,5,...,75,79  (== retrain_delay52 GATE_EPS)

NORM_BLOWUP = 10.0          # bounded weights: 10x init is a real blow-up (retrain_delay52 bar)
GABA_BLOWUP = 500.0         # W_msiInh2Exc_GABA ramps hugely by design; only >500x init is a true divergence
MSI_SNAP_COLLAPSE = 1e-3    # activity-collapse floor on the graph-available _g_out_sMSI snapshot
EP5_CLAMPFRAC_KILL = 0.05   # jitter-driven early GABA over-saturation (knob=0 baseline clampfrac_0=0.0000)

MON_W = ["W_inA", "W_inV", "W_MSI_exc", "W_MSI_inh", "W_a2msi_AMPA", "W_a2msi_NMDA",
         "W_v2msi_AMPA", "W_v2msi_NMDA"]
EXTRA_MON = ["W_msiInh2Exc_GABA"]


def P(log, *a):
    print(*a, flush=True)
    print(*a, file=log, flush=True)
    log.flush()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_net(seed):
    """CERTIFIED L6 recipe (retrain_delay52.build_delay52_net, verbatim) on the delay-corrected build."""
    net = PG.build_net(GDF, seed, batch_size=BS)
    net.plasticity_enabled = True
    net.enable_probe = False
    net._probe = None
    net._panel_enabled = False
    net._lever_L1_drop_inloop_log = False
    net._lever_L2_vector_reset    = True
    net._lever_L6_graph_train     = True
    net._lever_L4_graph           = True
    return net


def assert_reference(net, log):
    d_sub = int(getattr(net, "conduction_delay_msi_inh2exc"))
    assert d_sub == 52, f"FATAL: delay substeps={d_sub}, expected 52"
    assert abs(net.tau_gaba - TAU_GABA) < 1e-9, f"FATAL: tau_gaba={net.tau_gaba}, expected {TAU_GABA}"
    assert abs(float(net.sigma_dL_frames) - SIGMA_DL) < 1e-9, \
        f"FATAL: sigma_dL_frames={net.sigma_dL_frames}, expected {SIGMA_DL} (SIGMA_DL_FRAMES env not propagated)"
    _PV = float(os.environ.get('PV_GABA_SCALE', '1.0'))
    assert abs(float(net.pv_gaba_scale) - _PV) < 1e-9, \
        f"FATAL: pv_gaba_scale={net.pv_gaba_scale}, expected {_PV} (PV_GABA_SCALE env not propagated)"
    P(log, f"[id] delay={d_sub} substeps  tau_gaba={net.tau_gaba}  tau_nmda_inh={net.tau_nmda_inh}  "
           f"sigma_dL_frames={net.sigma_dL_frames}  pv_gaba_scale={net.pv_gaba_scale}  W_gaba_clamp={net.W_gaba_clamp}  rho0={net.rho0}")
    EXP_DCA = float(os.environ.get("DEND_COUPLING_ALPHA", 0.1))
    EXP_MGV = float(os.environ.get("MG_VHALF", -35.0))
    EXP_MGV_INH = float(os.environ.get("MG_VHALF_INH", EXP_MGV))
    EXP_MGK = float(os.environ.get("MG_K", 0.062))
    assert abs(float(net.dend_coupling_alpha) - EXP_DCA) < 1e-9, \
        f"FATAL: dend_coupling_alpha={net.dend_coupling_alpha}, expected {EXP_DCA} (DEND_COUPLING_ALPHA env not propagated)"
    assert abs(float(net.mg_vhalf) - EXP_MGV) < 1e-9, \
        f"FATAL: mg_vhalf={net.mg_vhalf}, expected {EXP_MGV} (MG_VHALF env not propagated)"
    assert abs(float(net.mg_vhalf_inh) - EXP_MGV_INH) < 1e-9, \
        f"FATAL: mg_vhalf_inh={net.mg_vhalf_inh}, expected {EXP_MGV_INH} (MG_VHALF_INH env not propagated / inh-gate decouple)"
    assert abs(float(net.mg_k) - EXP_MGK) < 1e-9, \
        f"FATAL: mg_k={net.mg_k}, expected {EXP_MGK} (MG_K env not propagated)"
    EXP_V2MSI = int(os.environ.get("CONDUCTION_DELAY_V2MSI", "400"))   # V-advance LATENCY lever (substeps); default 400 == byte-identical
    assert int(net.conduction_delay_v2msi) == EXP_V2MSI, \
        f"FATAL: conduction_delay_v2msi={net.conduction_delay_v2msi}, expected {EXP_V2MSI} (CONDUCTION_DELAY_V2MSI env not propagated / V-advance lever)"
    assert int(net.conduction_delay_v2msi_inh) == EXP_V2MSI + 20, \
        f"FATAL: conduction_delay_v2msi_inh={net.conduction_delay_v2msi_inh}, expected {EXP_V2MSI + 20} (inh leg must track v2msi+20)"
    EXP_GNMDA = float(os.environ.get("GNMDA", 1.30))
    assert abs(float(net.gNMDA) - EXP_GNMDA) < 1e-9, \
        f"FATAL: gNMDA={net.gNMDA}, expected {EXP_GNMDA} (GNMDA env; #66 dials C1 gain 1.30->0.70, gate KEPT open)"
    EXP_TAU_NMDA = float(os.environ.get("TAU_NMDA", "80.0"))
    assert abs(float(net.tau_nmda) - EXP_TAU_NMDA) < 1e-9, \
        f"FATAL: tau_nmda={net.tau_nmda}, expected {EXP_TAU_NMDA} (TAU_NMDA env not propagated / NR2A temporal lever)"
    EXP_G_REC = float(os.environ.get("G_REC", "0.1"))   # ep0 net.g_rec=0.0 (warmup); EXP_G_REC = post-ep25 operating point the loop applies
    assert EXP_G_REC >= 0.0, f"FATAL: G_REC env={EXP_G_REC} invalid (must be >=0)"
    P(log, f"[id] NR2A temporal: tau_nmda={float(net.tau_nmda)} (TAU_NMDA env)  g_rec_post_warmup={EXP_G_REC} (G_REC env; default 0.1)")
    EXP_MSIINH = float(os.environ.get("MSIINH_INPUT_SCALE", 1.0))
    assert abs(float(net.msiInh_input_init_scale) - EXP_MSIINH) < 1e-9, \
        f"FATAL: msiInh_input_init_scale={net.msiInh_input_init_scale}, expected {EXP_MSIINH} (MSIINH_INPUT_SCALE env not propagated)"
    EXP_GABA_SHUNT_SURR = (os.environ.get("GABA_SHUNT_SURR", "0") == "1")
    EXP_E_GABA = float(os.environ.get("E_GABA", -70.0))
    EXP_K_SHUNT_SURR = float(os.environ.get("K_SHUNT_SURR", 0.0))
    assert bool(net.gaba_shunt_surr) == EXP_GABA_SHUNT_SURR, \
        f"FATAL: gaba_shunt_surr={net.gaba_shunt_surr}, expected {EXP_GABA_SHUNT_SURR} (GABA_SHUNT_SURR env not propagated / surround-shunt)"
    assert abs(float(net.E_gaba) - EXP_E_GABA) < 1e-9, \
        f"FATAL: E_gaba={net.E_gaba}, expected {EXP_E_GABA} (E_GABA env not propagated)"
    assert abs(float(net.k_shunt_surr) - EXP_K_SHUNT_SURR) < 1e-9, \
        f"FATAL: k_shunt_surr={net.k_shunt_surr}, expected {EXP_K_SHUNT_SURR} (K_SHUNT_SURR env not propagated)"
    EXP_G_GABA = float(os.environ.get("G_GABA", "10"))
    assert abs(float(net.g_GABA) - EXP_G_GABA) < 1e-9, \
        f"FATAL: g_GABA={net.g_GABA}, expected {EXP_G_GABA} (G_GABA env not propagated / #92 inhibition lever)"
    EXP_MSIINH_GAIN = float(os.environ.get("MSIINH_INPUT_GAIN", 1.0))
    assert abs(float(net.msiInh_input_gain) - EXP_MSIINH_GAIN) < 1e-9, \
        f"FATAL: msiInh_input_gain={net.msiInh_input_gain}, expected {EXP_MSIINH_GAIN} (MSIINH_INPUT_GAIN env not propagated / task#102 (B) persistent FFI-recruitment forward-multiplier)"
    # task#104: STD/adaptation decouple knobs propagated into the TRAIN build (defaults byte-identical).
    for _attr, _envn, _dflt in [("tau_rec", "TAU_REC", 400.0), ("u_stp_a", "U_STP_A", 0.2),
                                ("u_stp_v", "U_STP_V", 0.2), ("nmda_std_scale", "NMDA_STD_SCALE", 0.8),
                                ("bM", "ADAPT_BM", 0.2), ("dM", "ADAPT_DM", 8.0),
                                ("aM", "ADAPT_A", 0.02),   # task#288 recovery-timescale lever (tau_adapt=1/aM); default 0.02 byte-identical
                                ("aMi", "ADAPT_A_INH", 0.1),   # task#376 interneuron SFA recovery-rate lever (tau_adapt_inh=1/aMi); default 0.1 byte-identical
                                ("dMi", "ADAPT_DM_INH", 2.0),  # task#376 interneuron spike-triggered adaptation increment; default 2.0 byte-identical
                                ("facil_gaba_on", "FACIL_GABA_ON", 0),        # task#385 SOM/Martinotti facilitation-in-time master switch; default 0 (OFF) byte-identical
                                ("tau_facil_gaba", "TAU_FACIL_GABA", 200.0),  # task#385 facilitation decay-to-baseline tau (ms); read only when ON
                                ("U_facil_gaba", "U_FACIL_GABA", 0.1),        # task#385 baseline utilization (near-zero-early); read only when ON

                                ("gAMPA_LP", "GAMPA_LP", 1.0),
                                ("v2msi_nmda_scale", "V2MSI_NMDA_SCALE", 1.0),   # latency #148: V->MSI NMDA fwd-scale (default 1.0 == byte-identical)
                                ("k_dvdt", "K_DVDT", 0.0),                # task#130 Form-A dVm/dt-adaptive threshold (Azouz&Gray 2000) latency fix
                                ("tau_dvdt", "TAU_DVDT", 3.0),
                                ("v_thresh_floor", "V_THRESH_FLOOR", 20.0),
                                ("dvdt_cap", "DVDT_CAP", 50.0)]:
        _exp = float(os.environ.get(_envn, _dflt))
        assert abs(float(getattr(net, _attr)) - _exp) < 1e-9, \
            f"FATAL: {_attr}={float(getattr(net, _attr))}, expected {_exp} ({_envn} env not propagated / task#104 STD-adaptation decouple knob)"
    # task#310: iSTDP inhibition-setpoint lever baked into TRAINING (active inline iSTDP subtracts istdp_baseline every step).
    EXP_ISTDP_BASE = float(os.environ.get("ISTDP_BASELINE", 0.6))
    assert abs(float(net.istdp_baseline) - EXP_ISTDP_BASE) < 1e-9, \
        f"FATAL: istdp_baseline={net.istdp_baseline}, expected {EXP_ISTDP_BASE} (ISTDP_BASELINE env not propagated / task#310 iSTDP inhibition-setpoint lever)"
    # task#327: independent per-channel afferent-arrival (conduction-delay) jitter σ (ms), baked into TRAINING.
    EXP_AFF_JIT = float(os.environ.get("AFFERENT_JITTER_MS", 0.0))
    assert abs(float(net.afferent_jitter_ms) - EXP_AFF_JIT) < 1e-9, \
        f"FATAL: afferent_jitter_ms={net.afferent_jitter_ms}, expected {EXP_AFF_JIT} (AFFERENT_JITTER_MS env not propagated / task#327 latency-shape lever)"
    aInh_norm = float(net.W_a2msiInh_AMPA.norm())
    P(log, f"[id] P5/C1 NMDA Mg-gate: dend_coupling_alpha={float(net.dend_coupling_alpha)} "
           f"(tau_eff={float(net.tau_m)/float(net.dend_coupling_alpha):.1f}ms)  mg_vhalf_exc={float(net.mg_vhalf)}  "
           f"mg_vhalf_inh={float(net.mg_vhalf_inh)}  mg_k={float(net.mg_k)}  gNMDA={float(net.gNMDA)}")
    P(log, f"[id] FF-inh lever: msiInh_input_init_scale={float(net.msiInh_input_init_scale)}  "
           f"msiInh_input_gain={float(net.msiInh_input_gain)} (B: persistent fwd-mult)  "
           f"|W_a2msiInh_AMPA|={aInh_norm:.4f}  (interneuron-input init scaled ~{EXP_MSIINH:.2f}x reference)")
    P(log, f"[id] STD/adapt decouple (task#104): tau_rec={float(net.tau_rec)}  u_stp_a={float(net.u_stp_a)}  "
           f"u_stp_v={float(net.u_stp_v)}  nmda_std_scale={float(net.nmda_std_scale)}  bM={float(net.bM)}  "
           f"dM={float(net.dM)}  aM={float(net.aM)}  (env-exposed; defaults 400/0.2/0.2/0.8/0.2/8.0/0.02 == byte-identical)")
    P(log, f"[id] iSTDP setpoint (task#310): istdp_baseline={float(net.istdp_baseline)}  eta_istdp={float(net.eta_istdp)}  "
           f"(ISTDP_BASELINE env; default 0.6 == byte-identical; lower baseline -> lower target rate -> settles HIGHER W_msiInh2Exc_GABA)")
    P(log, f"[id] afferent-arrival jitter (task#327): afferent_jitter_ms={float(net.afferent_jitter_ms)}  "
           f"(AFFERENT_JITTER_MS env; default 0.0 == byte-identical; per-channel INDEPENDENT σ on the A/V conduction-delay ring read -> graded first-spike-latency taper)")
    P(log, f"[id] latency #148 V->MSI NMDA fwd-scale: v2msi_nmda_scale={float(net.v2msi_nmda_scale)}  "
           f"(V2MSI_NMDA_SCALE env; default 1.0 == byte-identical; >1 strengthens slow voltage-dependent V NMDA for sub-threshold summation room)")
    P(log, f"[id] surround-shunt GABA: GABA_SHUNT_SURR={bool(net.gaba_shunt_surr)}  E_gaba={float(net.E_gaba)}  "
           f"k_shunt_surr={float(net.k_shunt_surr)}  (OFF default => byte-identical subtractive surround)")
    P(log, "[id] OK — delay52 / tau10 / common-mode ΔL fix is LIVE on the reference build")


def weight_norms(net):
    d = {}
    for nm in MON_W + EXTRA_MON:
        w = getattr(net, nm, None)
        if w is None:
            continue
        w = w.data if hasattr(w, "data") else w
        d[nm] = float(w.norm().item())
    return d


def all_weights_finite(net):
    bad = []
    for nm in MON_W + EXTRA_MON + ["_g_out_sA", "_g_out_sV", "_g_out_sMSI"]:
        w = getattr(net, nm, None)
        if w is None:
            continue
        w = w.data if hasattr(w, "data") else w
        if not bool(torch.isfinite(w).all().item()):
            bad.append(nm)
    return bad


def msi_rate(net):
    m = getattr(net, "_g_out_sMSI", None)
    return float(m.float().mean().item()) if m is not None else float("nan")


def msi_nactive(net):
    """Graph-available activity-breadth proxy: fraction of MSI-exc units with any snapshot spike."""
    m = getattr(net, "_g_out_sMSI", None)
    if m is None:
        return float("nan")
    return float((m.float() > 0).float().mean().item())


def gaba_clampfrac(net):
    g = net.W_msiInh2Exc_GABA.data
    return float((g >= 0.999 * net.W_gaba_clamp).float().mean().item())


def vitals_gate(net, ep, init_norms, hist, log):
    bad = all_weights_finite(net)
    norms = weight_norms(net)
    rate = msi_rate(net)
    nact = msi_nactive(net)
    cf = gaba_clampfrac(net)
    hist.setdefault("msi", []).append(rate)
    hist.setdefault("nactive", []).append(nact)
    hist.setdefault("clampfrac", []).append(cf)
    hist.setdefault("ep", []).append(ep)

    blown = [(nm, norms[nm], init_norms[nm]) for nm in MON_W
             if nm in norms and init_norms.get(nm, 0.0) > 0 and norms[nm] > NORM_BLOWUP * init_norms[nm]]
    for nm in EXTRA_MON:
        if nm in norms and init_norms.get(nm, 0.0) > 0 and norms[nm] > GABA_BLOWUP * init_norms[nm]:
            blown.append((nm, norms[nm], init_norms[nm]))

    hz = 1000.0 / net.dt
    P(log, f"[gate ep{ep:>2}] msi_snapshot={rate:.4e}/substep (~{rate*hz:.0f}Hz-equiv, inflated)  "
           f"n_active={nact:.3f}  gaba_clampfrac={cf:.4f}  finite={'OK' if not bad else 'BAD:'+','.join(bad)}")
    ratios = {nm: (norms[nm] / init_norms[nm] if init_norms.get(nm, 0.0) > 0 else float('nan')) for nm in norms}
    P(log, "[gate ep%2d] norm/init: " % ep + "  ".join(f"{nm}={ratios[nm]:.3f}x" for nm in (MON_W + EXTRA_MON) if nm in ratios))

    if bad:
        return False, f"NaN/Inf in {bad} at ep{ep}"
    # [#106 corrected gate] snapshot-collapse kill REMOVED (proven mis-calibrated): the <1e-3/substep
    # _g_out_sMSI proxy floor sits INSIDE the network's normal late-epoch low-rate band — the healthy
    # no-jitter reference also runs there (ep70 snapshot ~1.02e-3 -> 16.4 Hz canonical at ep79), so it
    # spuriously killed healthy seeds 42(ep60)/43(ep65). msi_snapshot + n_active stay computed/logged/
    # banked to hist (trajectory) above; the canonical-MSI sanity moves to the ep30 post-hoc eval path.
    if blown:
        return False, "weight-norm blow-up at ep%d: " % ep + "; ".join(
            f"{nm} {n:.3e} > x init {i:.3e}" for nm, n, i in blown)
    if ep == 5 and cf >= EP5_CLAMPFRAC_KILL:
        return False, (f"ep5 iSTDP-GABA over-saturation: clampfrac={cf:.4f} >= {EP5_CLAMPFRAC_KILL:.2f} "
                       f"(knob=0 baseline clampfrac_0=0.0000)")
    return True, "ok"


def main():
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log = open(LOG_PATH, "w", buffering=1)
    P(log, f"device={dev}  5-SEED RETRAIN (common-mode ΔL fix)  seed={SEED}  bs={BS}  n_seq={N_SEQ}  ep0..{N_EPOCHS-1}")
    P(log, f"GDF={GDF.__file__}   RUN_DIR={RUN_DIR}")
    P(log, f"single-variable change vs reference: SIGMA_DL_FRAMES={SIGMA_DL} (common-mode A-V ΔL ~N(0,{SIGMA_DL} frames))")
    P(log, f"schedule: g_rec=0.0 for ep<26, 0.1 for ep>=26   ckpts -> ep5/ep30/ep79")
    P(log, f"inline kills: NaN/Inf, norm>{NORM_BLOWUP}x init, ep5 GABA clampfrac>={EP5_CLAMPFRAC_KILL} "
           f"[#106 corrected gate: snapshot-collapse <{MSI_SNAP_COLLAPSE:.0e}/substep kill REMOVED — mis-calibrated]; "
           f"ep30 canonical-MSI + ep79 TBW/SBW/EI are POST-HOC")

    net = build_net(SEED)
    assert_reference(net, log)
    seed_all(SEED)
    init_norms = weight_norms(net)
    P(log, "[init] norms: " + "  ".join(f"{nm}={init_norms[nm]:.4e}" for nm in (MON_W + EXTRA_MON) if nm in init_norms))

    per_ep, ckpts, hist = [], {}, {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall0 = time.time()
    killed = None
    for ep in range(N_EPOCHS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        net.g_rec = (float(os.environ.get("G_REC", "0.1")) if ep > 25 else 0.0)   # G_REC env (default 0.1 == byte-identical)
        net.train_unsupervised_batch(N_SEQ, batch_size=BS, debug=False, epoch_idx=ep)
        net.W_MSI_exc.data.fill_diagonal_(0.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        per_ep.append(dt)
        if ep in GATE_EPS or ep < 2:
            P(log, f"[ep{ep:>2}] g_rec={net.g_rec:.1f}  train={dt:6.2f}s  cum={sum(per_ep)/60:5.2f}min")

        if ep in GATE_EPS:
            ok, msg = vitals_gate(net, ep, init_norms, hist, log)
            if not ok:
                killed = msg
                P(log, f"\n{'!'*64}\n[KILL] {msg}\n[KILL] stopping at ep{ep}; not saving further ckpts.\n{'!'*64}")
                if ep in CKPT_EP:
                    ck = GDF.make_checkpoint(net, epoch=ep, comment=f"retrain5 dL{SIG_TAG} seed{SEED} KILLED@{ep}")
                    torch.save(ck, CKPT_EP[ep]); ckpts[str(ep)] = CKPT_EP[ep]
                break

        if ep in CKPT_EP:
            ck = GDF.make_checkpoint(net, epoch=ep, comment=f"retrain5 dL{SIG_TAG} bs={BS} seed{SEED}")
            torch.save(ck, CKPT_EP[ep]); ckpts[str(ep)] = CKPT_EP[ep]
            P(log, f"[ep{ep:>2}] saved checkpoint {os.path.basename(CKPT_EP[ep])}")

    wall = time.time() - wall0
    fin_bad = all_weights_finite(net)
    P(log, f"\n{'='*64}")
    P(log, f"[RESULT] seed{SEED}: {'KILLED at ep%d: %s' % (len(per_ep)-1, killed) if killed else 'COMPLETE — %d epochs' % len(per_ep)}")
    P(log, f"[RESULT] wall={wall/60:.2f}min  end msi_rate={msi_rate(net):.4e}  n_active={msi_nactive(net):.3f}  "
           f"gaba_clampfrac={gaba_clampfrac(net):.4f}  finite={'OK' if not fin_bad else 'BAD:'+','.join(fin_bad)}")
    P(log, f"[RESULT] ckpts: {ckpts}")
    P(log, f"[RESULT] MSI snapshot trajectory: " +
           "  ".join(f"ep{e}={r:.3e}" for e, r in zip(hist.get('ep', []), hist.get('msi', []))))
    P(log, f"[RESULT] gaba clampfrac trajectory: " +
           "  ".join(f"ep{e}={c:.3f}" for e, c in zip(hist.get('ep', []), hist.get('clampfrac', []))))

    json.dump(dict(wall=wall, per_ep=per_ep, n_epochs_done=len(per_ep), killed=killed, ckpts=ckpts,
                   seed=SEED, bs=BS, n_seq=N_SEQ, sigma_dL_frames=SIGMA_DL, tau_gaba=float(net.tau_gaba),
                   msi_hist=hist.get('msi', []), nactive_hist=hist.get('nactive', []),
                   clampfrac_hist=hist.get('clampfrac', []), gate_eps=hist.get('ep', []),
                   final_finite=(not fin_bad)),
              open(JSON_PATH, "w"), indent=1)
    log.close()
    if killed or fin_bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
